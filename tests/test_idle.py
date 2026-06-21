from tokenwatt.meter import EnergyByRail, FakeMeter
from tokenwatt.idle import IdleBaseline


def test_idle_power_from_two_samples(monkeypatch):
    # cumulative advances 2 J on gpu each sample; we control the clock.
    meter = FakeMeter(cumulative_step=EnergyByRail({"gpu": 2.0}))
    clock = {"t": 0.0}
    import tokenwatt.idle as idle_mod
    monkeypatch.setattr(idle_mod.time, "monotonic", lambda: clock["t"])

    base = IdleBaseline(meter)
    base.sample()                 # t=0, cumulative gpu=2.0
    clock["t"] = 1.0
    base.sample()                 # t=1, cumulative gpu=4.0 -> 2.0 J over 1.0 s -> 2.0 W
    e = base.energy_over(3.0)     # 2.0 W * 3 s = 6.0 J
    assert e.joules["gpu"] == 6.0


def test_energy_over_zero_before_two_samples():
    base = IdleBaseline(FakeMeter())
    assert base.energy_over(5.0).total_j == 0.0


def test_in_flight_gate_skips_sample(monkeypatch):
    # Spec §7: a sample taken while a request is in-flight must be silently dropped.
    # With only ONE real sample recorded (the initial one), energy_over() returns 0.0
    # because two samples are required to compute a watt value.
    # The skipped mid-inference sample must NOT count as the second sample.
    meter = FakeMeter(cumulative_step=EnergyByRail({"gpu": 2.0}))
    clock = {"t": 0.0}
    import tokenwatt.idle as idle_mod
    monkeypatch.setattr(idle_mod.time, "monotonic", lambda: clock["t"])

    base = IdleBaseline(meter)
    base.sample()           # t=0 — first real sample recorded; not enough for watts yet

    clock["t"] = 1.0
    base.request_started()
    base.sample()           # must be skipped: in-flight counter is 1
    base.request_finished()

    # Only one real sample was recorded, so no delta was ever computed -> 0.0 J
    assert base.energy_over(1.0).total_j == 0.0


def test_energy_over_uses_median_not_last_sample(monkeypatch):
    # Three idle watt-readings [2, 2, 100] (one noisy spike). The baseline must use the
    # MEDIAN (2 W), not the most-recent reading (100 W), so a single spike can't skew it.
    clock = {"t": 0.0}
    import tokenwatt.idle as idle_mod
    monkeypatch.setattr(idle_mod.time, "monotonic", lambda: clock["t"])

    class ScriptedMeter:
        # cumulative GPU joules on each sample(); diffs over 1 s gaps -> [2, 2, 100] W
        def __init__(self):
            self._seq = iter([0.0, 2.0, 4.0, 104.0])

        def cumulative(self):
            return EnergyByRail({"gpu": next(self._seq)})

    base = IdleBaseline(ScriptedMeter())
    for _ in range(4):
        base.sample()
        clock["t"] += 1.0
    assert base.energy_over(1.0).joules["gpu"] == 2.0   # median(2, 2, 100) = 2, spike rejected


class _StepMeter:
    """Returns a preset cumulative value on each cumulative() call (J on the gpu rail)."""
    def __init__(self, vals):
        self._vals = vals; self._i = 0
    def cumulative(self):
        from tokenwatt.meter import EnergyByRail
        v = self._vals[min(self._i, len(self._vals) - 1)]; self._i += 1
        return EnergyByRail({"gpu": v})
    def begin(self, label): ...
    def end(self, label):
        from tokenwatt.meter import EnergyByRail
        return EnergyByRail({})


def test_sample_advances_cursor_while_busy_so_busy_span_is_discarded():
    m = _StepMeter([10.0, 1000.0, 1001.0])     # idle reading, busy reading, idle reading
    idle = IdleBaseline(m)
    idle.sample()                               # idle -> seeds _last = 10
    idle.request_started()
    idle.sample()                               # BUSY: must advance _last to 1000 and record nothing
    assert idle._last.joules["gpu"] == 1000.0   # fix: cursor advanced past the busy span
    assert len(idle._samples) == 0              # no idle sample recorded while busy
    idle.request_finished()
    idle.sample()                               # idle: delta = 1001-1000 = 1.0 (NOT 1001-10=991)
    assert len(idle._samples) == 1


def test_in_flight_accessor():
    idle = IdleBaseline(_StepMeter([0.0]))
    assert idle.in_flight() == 0
    idle.request_started(); idle.request_started()
    assert idle.in_flight() == 2
    idle.request_finished()
    assert idle.in_flight() == 1
