# tests/test_campaign.py
import pytest

import json as _json
import os as _os

from tokenwatt.meter import FakeMeter, EnergyByRail
from tokenwatt.metersource import FakeMeterSource
from tokenwatt import campaign


def _write_cfg(dirpath, quant):
    _os.makedirs(dirpath, exist_ok=True)
    doc = {"model_type": "x"}
    if quant is not None:
        doc["quantization"] = quant
    with open(_os.path.join(dirpath, "config.json"), "w") as f:
        _json.dump(doc, f)


def test_read_quantization_from_local_path(tmp_path):
    d = str(tmp_path / "mymodel")
    _write_cfg(d, {"group_size": 64, "bits": 4, "mode": "affine"})
    assert campaign.read_quantization(d) == {"bits": 4, "group_size": 64, "mode": "affine", "mixed": False}


def test_read_quantization_flags_mixed_precision(tmp_path):
    # a config with per-layer overrides (dict values) is mixed precision — the record must say so
    d = str(tmp_path / "mixed")
    _write_cfg(d, {"group_size": 32, "bits": 4, "mode": "mxfp4",
                   "model.layers.0.self_attn.q_proj": {"bits": 8, "mode": "affine"}})
    q = campaign.read_quantization(d)
    assert q["bits"] == 4 and q["mode"] == "mxfp4" and q["mixed"] is True


def test_read_quantization_from_hf_cache(tmp_path):
    # emulate the HF hub layout: <hub>/models--{org}--{name}/snapshots/<hash>/config.json
    snap = str(tmp_path / "hub" / "models--mlx-community--Foo-8bit" / "snapshots" / "abc")
    _write_cfg(snap, {"group_size": 64, "bits": 8, "mode": "affine"})
    q = campaign.read_quantization("mlx-community/Foo-8bit", hf_home=str(tmp_path))
    assert q == {"bits": 8, "group_size": 64, "mode": "affine", "mixed": False}


def test_read_quantization_none_when_config_missing(tmp_path):
    # honest "unknown" (None), never a fabricated value, when nothing resolves
    assert campaign.read_quantization("mlx-community/Nope", hf_home=str(tmp_path)) is None


def test_read_quantization_full_precision_when_no_quant_block(tmp_path):
    d = str(tmp_path / "fp")
    _write_cfg(d, None)   # config present but no quantization key = full precision, not unknown
    assert campaign.read_quantization(d) == {"bits": None, "group_size": None, "mode": "none", "mixed": False}


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


from tokenwatt.meter import FakeMeter


def _ok_source(*_a, **_k):
    s = FakeMeterSource([1.0, 1.0, 1.001])
    s.reachable = lambda: (True, "fake ok")   # type: ignore[attr-defined]
    return s


def _dead_source(*_a, **_k):
    s = FakeMeterSource([1.0])
    s.reachable = lambda: (False, "ConnectError: no route")  # type: ignore[attr-defined]
    return s


def test_run_probe_returns_result_with_injected_fakes():
    clk = _Clock()
    result, msg = campaign.run_probe(
        host="h", seconds=1.0, poll_s=0.5,
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=_ok_source, sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert result.rail_total_j > 0
    assert "fake ok" in msg


def test_run_probe_aborts_loud_when_meter_unreachable():
    result, msg = campaign.run_probe(host="h", make_source=_dead_source,
                                     make_meter=lambda: FakeMeter())
    assert result is None                      # fail loud, no probe attempted
    assert "unreachable" in msg.lower()


def test_run_probe_degrades_when_rail_meter_unavailable():
    def _boom():
        raise RuntimeError("not Apple Silicon")
    result, msg = campaign.run_probe(host="h", make_source=_ok_source, make_meter=_boom)
    assert result is None
    assert "meter" in msg.lower()


def test_format_probe_hints_when_cadence_not_observed():
    # window was too short to see the plug's accumulator change >=2 times: resolution/cadence
    # come back None. Printing bare dashes next to "pick C1 cell length well above this" is
    # self-contradictory (there's nothing to compare against) — must be an actionable hint
    # instead, and the "pick C1 cell length" phrasing must be gone (there is no measured cadence
    # to pick above).
    r = campaign.ProbeResult(
        dt_s=2.0, wall_wh=0.0, wall_w=0.0, rail_total_j=40.0, rail_w=20.0,
        rail_by_rail_j={"cpu_total": 40.0}, ratio_wall_over_rail=None,
        meter_resolution_wh=None, meter_cadence_s=None, n_samples=5)
    out = campaign.format_probe(r)
    assert "not observed" in out
    assert "larger --seconds" in out
    assert "pick C1 cell length" not in out
    # honesty boundary still holds even in the under-window case
    assert "calibrated" not in out.lower()
