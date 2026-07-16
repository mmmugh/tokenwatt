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


def test_save_refuses_unnamed_model(tmp_path):
    # a campaign whose served model id is unknown ("?") must NOT be written as a
    # degenerate "<machine>__.json" keyed profile — that pollutes the keyed
    # namespace and would later be handed back for who-knows-which model. Fail loud
    # and write nothing.
    with pytest.raises(ValueError, match="unnamed model"):
        save(_profile("?"), root=str(tmp_path))
    assert not os.listdir(str(tmp_path))               # nothing degenerate left behind


def test_load_ignores_a_degenerate_keyed_profile(tmp_path):
    # defense against a "<machine>__.json" left by older code: a request for an
    # unnamed model ("?") must not resolve to it — an unknown model cannot be
    # honestly matched to any calibration, so load() returns None.
    from dataclasses import asdict
    p = _profile("?")                                  # model_calibrated_on == "?"
    d = asdict(p)
    open(os.path.join(tmp_path, f"{p.machine_id}__.json"), "w").write(json.dumps(d))
    assert load(p.machine_id, "?", root=str(tmp_path)) is None


def test_load_unnamed_model_ignores_legacy_file(tmp_path):
    # same honesty gap on the LEGACY path: a pre-schema-2 "<machine>.json" whose model
    # is the "?" sentinel must NOT be handed back for a request for an unnamed model.
    # load() returns None for an unnamed model on EVERY path, keyed or legacy.
    from dataclasses import asdict
    p = _profile("?")                                  # model_calibrated_on == "?"
    d = asdict(p)
    open(os.path.join(tmp_path, f"{p.machine_id}.json"), "w").write(json.dumps(d))
    assert load(p.machine_id, "?", root=str(tmp_path)) is None


def test_profile_schema_bumped_to_3_and_schema_2_still_loads(tmp_path):
    # adding quantization bumps the profile schema to 3 so old code reading a new file fails loud
    # ('unknown schema') rather than a raw TypeError; schema-2 files still load (back-compat).
    from dataclasses import asdict
    p = _profile("qwen3.6-27b")
    d = asdict(p)
    assert d["schema_version"] == 3
    d2 = dict(d); d2["schema_version"] = 2; d2.pop("quantization")   # a pre-quantization schema-2 file
    open(os.path.join(tmp_path, f"{p.machine_id}__qwen3-6-27b.json"), "w").write(json.dumps(d2))
    got = load(p.machine_id, "qwen3.6-27b", root=str(tmp_path))
    assert got is not None and got.quantization is None


def test_profile_carries_quantization(tmp_path):
    # the fitted profile records the served model's quantization and round-trips it
    p = profile_from(_fit(), _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0,
                     quantization={"bits": 8, "group_size": 64, "mode": "affine", "mixed": False})
    assert p.quantization == {"bits": 8, "group_size": 64, "mode": "affine", "mixed": False}
    save(p, root=str(tmp_path))
    got = load(p.machine_id, "qwen3.6-27b", root=str(tmp_path))
    assert got.quantization == {"bits": 8, "group_size": 64, "mode": "affine", "mixed": False}


def test_profile_without_quantization_defaults_to_none(tmp_path):
    # a profile built/written before quantization existed loads with None, not a fabricated value
    from dataclasses import asdict
    p = _profile("qwen3.6-27b")                        # profile_from with no quantization
    assert p.quantization is None
    d = asdict(p); d.pop("quantization")               # simulate a pre-quant profile file
    open(os.path.join(tmp_path, f"{p.machine_id}__qwen3-6-27b.json"), "w").write(json.dumps(d))
    got = load(p.machine_id, "qwen3.6-27b", root=str(tmp_path))
    assert got is not None and got.quantization is None


def test_list_profiles_returns_legacy_schema1_profile(tmp_path):
    # list_profiles routes through _read, which accepts schema 1 and 2, so a legacy
    # single-model "<machine>.json" (schema 1) is still surfaced, not dropped.
    from dataclasses import asdict
    legacy = _profile("qwen3.6-27b")
    d = asdict(legacy); d["schema_version"] = 1
    open(os.path.join(tmp_path, f"{legacy.machine_id}.json"), "w").write(json.dumps(d))
    profs = list_profiles(root=str(tmp_path))
    assert len(profs) == 1 and profs[0].model_calibrated_on == "qwen3.6-27b"


def test_list_profiles_fails_loud_on_unknown_schema(tmp_path):
    # list_profiles must validate each file through the SAME fail-loud reader as
    # load(): a profile with an unrecognized schema is a real problem to surface,
    # not something to hand back as a silently-constructed Profile (or crash with a
    # bare TypeError on the missing fields).
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    bad = os.path.join(tmp_path, "mac15-14_apple-m3-ultra_96gb_macos26__gpt-oss.json")
    open(bad, "w").write(json.dumps({"schema_version": 99, "machine_id": "x"}))
    with pytest.raises(ValueError, match="schema"):
        list_profiles(root=str(tmp_path))


def test_profile_persists_n_excluded_from_fit(tmp_path):
    # the count of battery-contaminated cells dropped from the fit is provenance:
    # a profile fit with cells excluded is less trustworthy, so the number must
    # survive to disk (not just print to the console) and round-trip on load.
    fit = FitResult(fit_type="scalar", a=1.83, b=0.4, residual_rel=0.03,
                    run_variance_rel=0.02, band_pct=4.0, tier="plug-calibrated (±4%)",
                    n_samples=6, n_passes=2, n_excluded=3)
    p = profile_from(fit, _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0)
    assert p.n_excluded == 3
    save(p, root=str(tmp_path))
    got = load(p.machine_id, "qwen3.6-27b", root=str(tmp_path))
    assert got.n_excluded == 3                         # round-tripped through disk


def test_profile_without_n_excluded_defaults_to_none(tmp_path):
    # a profile written before n_excluded existed (no such key on disk) has UNKNOWN
    # provenance. Per the honesty contract an absent field is None, never a fabricated
    # "0 cells excluded" (which would read as fully-trustworthy). It must load as None.
    from dataclasses import asdict
    p = _profile("qwen3.6-27b")
    d = asdict(p); d.pop("n_excluded")                 # simulate a pre-M5 profile file
    open(os.path.join(tmp_path, f"{p.machine_id}__qwen3-6-27b.json"), "w").write(json.dumps(d))
    got = load(p.machine_id, "qwen3.6-27b", root=str(tmp_path))
    assert got is not None and got.n_excluded is None


def test_keyed_load_rejects_slug_collision_wrong_model(tmp_path):
    # "qwen3-5-4b" and "qwen3.5-4b" both slug to "qwen3-5-4b", so they share one keyed file.
    # A stored profile calibrated on "qwen3-5-4b" must NOT be handed back for a request for
    # "qwen3.5-4b" just because the filename matches -- that would silently return a
    # wrong-model plug-calibrated band, which the honesty contract forbids.
    save(_profile("qwen3-5-4b"), root=str(tmp_path))
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.5-4b", root=str(tmp_path)) is None
