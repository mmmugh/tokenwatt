import json
import os

import pytest

from tokenwatt.meter import FakeMeter, EnergyByRail
from tokenwatt.metersource import FakeMeterSource
from tokenwatt.battery import ChatResult, FakeLoadClient, LoadCell
from tokenwatt.powersource import BatteryFlux, FakeBatterySource
from tokenwatt import campaign


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_measure_idle_returns_per_rail_and_wall_watts():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))  # +10 J/read
    source = FakeMeterSource([100.0, 100.05])                            # +0.05 Wh over window
    idle = campaign.measure_idle(meter, source, seconds=10.0,
                                 sleep=clk.sleep, monotonic=clk.monotonic)
    assert idle.dt_s == pytest.approx(10.0)
    assert idle.rail_w == {"cpu_total": pytest.approx(1.0)}      # 10 J / 10 s
    assert idle.wall_w == pytest.approx(0.05 * 3600 / 10.0)      # 18 W


def test_measure_idle_records_battery_abs_w():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.0, 100.05])
    idle = campaign.measure_idle(meter, source, seconds=10.0,
                                 battery=FakeBatterySource([BatteryFlux(9.0, True)]),
                                 sleep=clk.sleep, monotonic=clk.monotonic)
    assert idle.battery_abs_w == pytest.approx(9.0)


def test_measure_idle_catches_a_mid_window_battery_spike():
    # a charge burst that begins AND ends between the boundary reads is invisible
    # if idle only samples the battery at the window's start and end. Sampling
    # across the window (like run_cell) must catch the peak, so a contaminated idle
    # baseline is gated instead of silently poisoning the marginal subtraction.
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.0, 100.05])
    # low at the boundaries; a 50 W spike only in the middle of the window
    battery = FakeBatterySource([BatteryFlux(1.0, False), BatteryFlux(1.0, False),
                                 BatteryFlux(50.0, True), BatteryFlux(1.0, False)])
    idle = campaign.measure_idle(meter, source, seconds=15.0, battery=battery,
                                 sleep=clk.sleep, monotonic=clk.monotonic)
    assert idle.battery_abs_w == pytest.approx(50.0)      # spike caught mid-window
    assert idle.dt_s == pytest.approx(15.0)


def test_run_cell_brackets_load_and_subtracts_idle_from_both_sides():
    clk = _Clock()
    # rail rises 10 J between the two cumulative() reads (start, end)
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.000, 100.100])                # +0.1 Wh -> 360 J wall
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.5}, wall_w=3.0, dt_s=10.0)
    # each load call advances the clock 2 s and returns 7 in / 11 out tokens
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=7, tok_out=11)

    s = campaign.run_cell(meter, source, load_fn, idle, cell="prefill", model="m1",
                          seconds=10.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.cell == "prefill" and s.model == "m1"
    assert s.dt_s == pytest.approx(10.0)
    # 5 load calls fit in 10 s at 2 s each
    assert (s.tok_in, s.tok_out) == (35, 55)
    assert s.requests == 5                          # always-known load-validity signal
    # marginal rail = ΔE_rail − idle_rail·dt = 10 − 0.5·10 = 5 J
    assert s.e_rail_marginal_j == {"cpu_total": pytest.approx(5.0)}
    # marginal wall = ΔWh·3600 − idle_wall·dt = 360 − 3·10 = 330 J
    assert s.e_wall_marginal_j == pytest.approx(330.0)


def test_run_cell_records_tokens_unknown_but_still_counts_requests():
    # an upstream that withholds usage -> tok_in/out None (never a fake 0), but the
    # request count still proves sustained load ran (the efficacy signal)
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([1.0, 1.1])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=None, tok_out=None)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="decode", model="m",
                          seconds=6.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.tok_in is None and s.tok_out is None    # honest unknown, not 0
    assert s.requests == 3                            # load still ran and was counted
    assert s.e_wall_marginal_j >= 0.0                # energy sample still valid


def test_run_cell_clamps_negative_marginals_to_zero():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 1.0}))   # tiny rail move
    source = FakeMeterSource([50.0, 50.0])                                # wall flat
    idle = campaign.IdleRates(rail_w={"cpu_total": 100.0}, wall_w=100.0, dt_s=1.0)
    def load_fn():
        clk.sleep(1.0)
        return ChatResult(tok_in=1, tok_out=1)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="decode", model="m",
                          seconds=1.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.e_rail_marginal_j["cpu_total"] == 0.0     # never a negative energy
    assert s.e_wall_marginal_j == 0.0


def test_run_cell_records_max_battery_abs_w():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.0, 100.1])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    battery = FakeBatterySource([BatteryFlux(3.0, True), BatteryFlux(21.0, True)])  # peak 21 W
    def load_fn():
        clk.sleep(6.0)                                   # each call crosses a 5 s battery poll
        return ChatResult(tok_in=1, tok_out=1)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="prefill", model="m",
                          seconds=12.0, battery=battery, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.battery_abs_w == pytest.approx(21.0)         # the worst contamination in the cell


def test_run_cell_battery_none_when_no_source_or_desktop():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([1.0, 1.0])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=1, tok_out=1)
    # no source at all
    s1 = campaign.run_cell(meter, source, load_fn, idle, cell="c", model="m",
                           seconds=2.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s1.battery_abs_w is None
    # desktop: source present but always yields None
    clk2 = _Clock()
    def load_fn2():
        clk2.sleep(2.0)
        return ChatResult(tok_in=1, tok_out=1)
    s2 = campaign.run_cell(meter, FakeMeterSource([1.0, 1.0]), load_fn2, idle, cell="c",
                           model="m", seconds=2.0, battery=FakeBatterySource([None]),
                           sleep=clk2.sleep, monotonic=clk2.monotonic)
    assert s2.battery_abs_w is None


def test_run_cell_zero_seconds_records_unknown_tokens_not_zero():
    # a zero/negative `seconds` window runs no requests at all -> requests == 0.
    # any_unknown never flips True in that case, so without the fix tok_in/tok_out
    # would fall through to the fabricated `0` initializer instead of an honest None.
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([1.0, 1.0])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=7, tok_out=11)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="prefill", model="m",
                          seconds=0.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.requests == 0
    assert s.tok_in is None and s.tok_out is None    # never a fabricated 0


def _cells():
    return [LoadCell("prefill", "p", 8), LoadCell("decode", "d", 1024)]


class _RaisingLoad:
    """A load client whose `chat` always fails mid-cell, to exercise the
    named-failure-phase path in run_campaign."""
    def chat(self, model, prompt, max_tokens):
        raise RuntimeError("timeout")


def test_run_campaign_collects_one_sample_per_cell_per_pass():
    clk = _Clock()
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([None]),
        host="h", cell_seconds=4.0, passes=2, timestamp=1_700_000_000.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert len(result.samples) == 4                       # 2 cells × 2 passes
    assert [s.cell for s in result.samples] == ["prefill", "decode", "prefill", "decode"]
    assert result.model == "m1" and result.passes == 2 and result.cell_seconds == 4.0
    assert result.meter_tier == "fake"                    # from the FakeMeterSource
    assert all(s.tok_in > 0 for s in result.samples)      # load actually ran


def test_run_campaign_idle_seconds_sizes_idle_window_independently():
    # WHY: on laptops the plug's coarse 0.119 Wh quantum wrecks a short idle
    # baseline. --idle-seconds must let the idle window be sized independently of
    # (and longer than) the per-cell duration; without it, idle == cell_seconds.
    clk = _Clock()
    result, _ = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([None]),
        host="h", cell_seconds=4.0, idle_seconds=20.0, passes=1, timestamp=0.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert result.idle.dt_s == pytest.approx(20.0)              # idle used idle_seconds…
    assert all(s.dt_s == pytest.approx(4.0) for s in result.samples)  # …cells used cell_seconds


def test_run_campaign_idle_seconds_defaults_to_cell_seconds():
    # WHY: omitting --idle-seconds must preserve the prior behavior exactly
    # (idle window == cell_seconds), so desktop/Ultra campaigns are byte-unchanged.
    clk = _Clock()
    result, _ = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([None]),
        host="h", cell_seconds=7.0, passes=1, timestamp=0.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert result.idle.dt_s == pytest.approx(7.0)               # defaulted to cell_seconds


def test_run_campaign_threads_battery_into_samples():
    clk = _Clock()
    result, _ = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([BatteryFlux(4.0, True)]),
        host="h", cell_seconds=4.0, passes=1, timestamp=0.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert all(s.battery_abs_w == pytest.approx(4.0) for s in result.samples)
    assert result.idle.battery_abs_w == pytest.approx(4.0)


def test_run_campaign_aborts_loud_when_meter_source_unreachable():
    def _dead_source(*_a, **_k):
        s = FakeMeterSource([1.0])
        s.reachable = lambda: (False, "ConnectError")   # type: ignore[attr-defined]
        return s
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m", load=FakeLoadClient(),
        make_meter=lambda: FakeMeter(), make_source=_dead_source,
        host="h", timestamp=0.0)
    assert result is None and "unreachable" in msg.lower()


def test_run_campaign_aborts_loud_when_meter_unavailable():
    def _boom():
        raise RuntimeError("no zeus")
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m", load=FakeLoadClient(),
        make_meter=_boom, make_source=lambda *_a, **_k: FakeMeterSource([1.0]),
        host="h", timestamp=0.0)
    assert result is None
    assert "meter unavailable" in msg.lower()


def test_run_campaign_read_failure_names_the_phase():
    clk = _Clock()
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m", load=_RaisingLoad(),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([None]),
        host="h", cell_seconds=4.0, passes=1, timestamp=0.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is None
    assert "campaign read failed during" in msg
    assert "prefill" in msg                          # names the failing cell, not just the type


def test_run_campaign_emits_progress_per_phase():
    clk = _Clock()
    events: list[str] = []
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([None]),
        host="h", cell_seconds=4.0, passes=1, timestamp=0.0,
        on_progress=events.append,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert len(events) >= 3                           # idle + 2 cells
    assert any("idle" in e.lower() for e in events)
    assert any("prefill" in e for e in events)
    assert any("decode" in e for e in events)


def test_write_campaign_round_trips_json(tmp_path):
    clk = _Clock()
    result, _ = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([None]),
        host="h", cell_seconds=4.0, passes=1, timestamp=1_700_000_000.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    out = os.path.join(tmp_path, "sub", "campaign.json")
    campaign.write_campaign(result, out)
    doc = json.load(open(out))
    assert doc["model"] == "m1"
    assert doc["meter"]["tier"] == "fake"
    assert len(doc["samples"]) == 2
    assert doc["samples"][0]["cell"] == "prefill"
    assert "e_wall_marginal_j" in doc["samples"][0]
    assert "requests" in doc["samples"][0]        # efficacy signal persisted for C2
    assert "battery_abs_w" in doc["samples"][0]
    assert "wall_w" in doc["idle"]
    assert "battery_abs_w" in doc["idle"]
