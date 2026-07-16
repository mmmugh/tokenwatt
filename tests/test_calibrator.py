# tests/test_calibrator.py
from tokenwatt import profiles
from tokenwatt.calibration import tier_label
from tokenwatt.calibrator import Calibrator
from tokenwatt.profiles import Profile


def _profile(machine_id, model, *, a, b, band_pct, meter_tier="smart_plug"):
    return Profile(
        machine_id=machine_id, label="test machine", fit_type="scalar",
        coefficients={"a": a, "b": b}, residual_rel=0.02, run_variance_rel=0.01,
        band_pct=band_pct, tier=tier_label(meter_tier, band_pct),
        meter={"name": "shelly", "tier": meter_tier, "accuracy_pct": 1.0},
        n_samples=6, n_passes=2, model_calibrated_on=model, macos="26", created_at=1000.0,
    )


def test_applies_a_certified_profile(tmp_path):
    root = str(tmp_path)
    profiles.save(_profile("mach1", "qwen3.6-27b", a=2.0, b=0.0, band_pct=2.7), root=root)
    cal = Calibrator("mach1", root=root)
    applied = cal.apply("qwen3.6-27b", marginal_rail_j=1000.0, dt_s=300.0)
    assert applied is not None
    assert applied.wall_j == 2000.0                       # 2.0·1000 + 0·300
    assert applied.tier.startswith("plug-calibrated")
    assert applied.band_pct == 2.7


def test_b_term_scales_with_request_duration(tmp_path):
    root = str(tmp_path)
    profiles.save(_profile("mach1", "m", a=1.0, b=0.5, band_pct=3.0), root=root)
    applied = Calibrator("mach1", root=root).apply("m", marginal_rail_j=1000.0, dt_s=300.0)
    assert applied.wall_j == 1000.0 + 0.5 * 300.0         # a·rail + b·Δt = 1150


def test_uncertified_profile_is_not_applied(tmp_path):
    # a plug fit no tighter than the estimated floor must NOT re-price a live cost
    # (honesty contract: never dress up an uncertified number as calibrated)
    root = str(tmp_path)
    profiles.save(_profile("mach1", "wide", a=2.0, b=0.0, band_pct=25.0), root=root)
    cal = Calibrator("mach1", root=root)
    assert cal.profile_for("wide") is None
    assert cal.apply("wide", 1000.0, 300.0) is None


def test_no_profile_for_unknown_model(tmp_path):
    assert Calibrator("mach1", root=str(tmp_path)).apply("never-calibrated", 1000.0, 300.0) is None


def test_profile_for_a_different_machine_is_ignored(tmp_path):
    root = str(tmp_path)
    profiles.save(_profile("other-machine", "m", a=2.0, b=0.0, band_pct=2.7), root=root)
    assert Calibrator("mach1", root=root).apply("m", 1000.0, 300.0) is None


def test_available_models_lists_only_this_machines_certified(tmp_path):
    root = str(tmp_path)
    profiles.save(_profile("mach1", "good", a=1.0, b=0.0, band_pct=3.0), root=root)
    profiles.save(_profile("mach1", "wide", a=1.0, b=0.0, band_pct=25.0), root=root)   # uncertified
    profiles.save(_profile("other", "good", a=1.0, b=0.0, band_pct=3.0), root=root)    # wrong machine
    assert Calibrator("mach1", root=root).available_models() == ["good"]


def test_corrupt_profile_is_treated_as_estimated_not_raised(tmp_path):
    # a truncated/corrupt profile (e.g. an interrupted `calibrate fit`) must NOT raise into the
    # request path: the model is priced as estimated, and the miss is cached (logged once).
    import os
    root = str(tmp_path)
    with open(os.path.join(root, "mach1__m.json"), "w") as f:
        f.write('{ "machine_id": "mach1", "model_calibrated_on": "m",')      # truncated JSON
    cal = Calibrator("mach1", root=root)
    assert cal.profile_for("m") is None                                      # guarded, no exception
    assert cal.apply("m", 1000.0, 300.0) is None
    assert "m" in cal._cache                                                 # miss cached, won't re-read


def test_profile_is_read_once_and_cached(tmp_path):
    root = str(tmp_path)
    profiles.save(_profile("mach1", "m", a=2.0, b=0.0, band_pct=2.7), root=root)
    cal = Calibrator("mach1", root=root)
    assert cal.apply("m", 1000.0, 300.0).wall_j == 2000.0
    # deleting the file after first use must NOT change the cached answer (proves caching:
    # recalibration takes effect on the next serve, not mid-process)
    import os
    for f in os.listdir(root):
        os.remove(os.path.join(root, f))
    assert cal.apply("m", 1000.0, 300.0).wall_j == 2000.0     # still applies from cache
