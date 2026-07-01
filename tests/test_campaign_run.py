import pytest

from tokenwatt.meter import FakeMeter, EnergyByRail
from tokenwatt.metersource import FakeMeterSource
from tokenwatt.battery import ChatResult
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
