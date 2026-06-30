# tests/test_campaign.py
import pytest

from tokenwatt.meter import FakeMeter, EnergyByRail
from tokenwatt.metersource import FakeMeterSource
from tokenwatt import campaign


class _Clock:
    """Deterministic clock: sleep() advances virtual time; monotonic() reads it."""
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_probe_computes_synchronized_deltas_and_meter_characterization():
    clk = _Clock()
    # one rail rises 10 J per cumulative() read; 5 reads over the window -> 40 J delta
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    # accumulator (Wh) ticks by 0.001 Wh every other poll -> resolution 0.001, cadence 1.0s
    source = FakeMeterSource([100.000, 100.000, 100.001, 100.001, 100.002])

    r = campaign.probe(meter, source, seconds=2.0, poll_s=0.5,
                       sleep=clk.sleep, monotonic=clk.monotonic)

    assert r.n_samples == 5
    assert r.dt_s == pytest.approx(2.0)
    assert r.wall_wh == pytest.approx(0.002, abs=1e-9)
    assert r.wall_w == pytest.approx(0.002 * 3600 / 2.0)        # J/s over the window
    assert r.rail_total_j == pytest.approx(40.0)
    assert r.rail_by_rail_j == {"cpu_total": pytest.approx(40.0)}
    assert r.rail_w == pytest.approx(20.0)
    assert r.ratio_wall_over_rail == pytest.approx(0.002 * 3600 / 40.0)
    # the C0 raison d'être: characterize the plug before trusting it
    assert r.meter_resolution_wh == pytest.approx(0.001, abs=1e-9)
    assert r.meter_cadence_s == pytest.approx(1.0)


def test_probe_ratio_is_none_when_no_rail_movement():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({}))           # rails flat
    source = FakeMeterSource([5.0, 5.0, 5.0])
    r = campaign.probe(meter, source, seconds=1.0, poll_s=0.5,
                       sleep=clk.sleep, monotonic=clk.monotonic)
    assert r.rail_total_j == 0.0
    assert r.ratio_wall_over_rail is None     # never divide-by-zero into a fake ratio


def test_format_probe_is_plain_and_marks_uncalibrated():
    r = campaign.ProbeResult(
        dt_s=2.0, wall_wh=0.002, wall_w=3.6, rail_total_j=40.0, rail_w=20.0,
        rail_by_rail_j={"cpu_total": 40.0}, ratio_wall_over_rail=0.18,
        meter_resolution_wh=0.001, meter_cadence_s=1.0, n_samples=5)
    out = campaign.format_probe(r)
    assert "wall" in out.lower() and "rail" in out.lower()
    assert "cpu_total" in out
    # must not masquerade as a calibration
    assert "calibrated" not in out.lower()
