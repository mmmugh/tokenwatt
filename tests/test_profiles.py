# tests/test_profiles.py
import json
import os
import pytest

from tokenwatt.profiles import Profile, profile_from, save, load, active_for, list_profiles
from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo


def _fit():
    return FitResult(fit_type="scalar", a=1.83, b=0.4, residual_rel=0.03,
                     run_variance_rel=0.02, band_pct=4.0, tier="plug-calibrated (±4%)",
                     n_samples=6, n_passes=2)


def _machine():
    return MachineInfo("mac15-14_apple-m3-ultra_96gb_macos26", "Apple M3 Ultra · 96 GB",
                       "Apple M3 Ultra", "Mac15,14", 96, "26")


def _profile(model):
    return profile_from(_fit(), _machine(),
                        meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                        model_calibrated_on=model, created_at=1_700_000_000.0)


def test_profile_from_carries_fit_machine_and_meter():
    p = profile_from(_fit(), _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0)
    assert p.machine_id == "mac15-14_apple-m3-ultra_96gb_macos26"
    assert p.coefficients == {"a": 1.83, "b": 0.4}
    assert p.tier == "plug-calibrated (±4%)"
    assert p.meter["tier"] == "smart_plug"
    assert p.model_calibrated_on == "qwen3.6-27b"


def test_save_keys_by_machine_and_model(tmp_path):
    path = save(_profile("qwen3.6-27b"), root=str(tmp_path))
    assert path.endswith("mac15-14_apple-m3-ultra_96gb_macos26__qwen3-6-27b.json")
    got = load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path))
    assert got == _profile("qwen3.6-27b")


def test_two_models_on_one_machine_coexist(tmp_path):
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    save(_profile("gemma-4-e4b"), root=str(tmp_path))
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path)) is not None
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "gemma-4-e4b", root=str(tmp_path)) is not None
    assert len(list_profiles(root=str(tmp_path))) == 2


def test_load_missing_model_returns_none(tmp_path):
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "other-model", root=str(tmp_path)) is None
    assert load("no-such-machine", "qwen3.6-27b", root=str(tmp_path)) is None


def test_legacy_schema1_file_loads_only_for_its_model(tmp_path):
    # a pre-existing <machine>.json (schema 1) still resolves for its own model
    legacy = _profile("qwen3.6-27b")
    from dataclasses import asdict
    d = asdict(legacy); d["schema_version"] = 1
    open(os.path.join(tmp_path, f"{legacy.machine_id}.json"), "w").write(json.dumps(d))
    assert load(legacy.machine_id, "qwen3.6-27b", root=str(tmp_path)) is not None
    assert load(legacy.machine_id, "different-model", root=str(tmp_path)) is None   # not a blind match


def test_load_raises_on_unknown_schema(tmp_path):
    p = save(_profile("qwen3.6-27b"), root=str(tmp_path))
    d = json.load(open(p)); d["schema_version"] = 99
    open(p, "w").write(json.dumps(d))
    with pytest.raises(ValueError, match="schema"):
        load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path))


def test_active_for_is_exact_model_match(tmp_path):
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    assert active_for("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path)) is not None
    assert active_for("mac15-14_apple-m3-ultra_96gb_macos26", "gpt-oss-120b", root=str(tmp_path)) is None


def test_keyed_load_rejects_slug_collision_wrong_model(tmp_path):
    # "qwen3-5-4b" and "qwen3.5-4b" both slug to "qwen3-5-4b", so they share one keyed file.
    # A stored profile calibrated on "qwen3-5-4b" must NOT be handed back for a request for
    # "qwen3.5-4b" just because the filename matches -- that would silently return a
    # wrong-model plug-calibrated band, which the honesty contract forbids.
    save(_profile("qwen3-5-4b"), root=str(tmp_path))
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.5-4b", root=str(tmp_path)) is None
