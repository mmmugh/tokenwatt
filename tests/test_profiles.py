# tests/test_profiles.py
import os

from tokenwatt.profiles import Profile, profile_from, save, load, list_profiles
from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo


def _fit():
    return FitResult(fit_type="scalar", a=1.83, b=0.4, residual_rel=0.03,
                     run_variance_rel=0.02, band_pct=4.0, tier="plug-calibrated (±4%)",
                     n_samples=6, n_passes=2)


def _machine():
    return MachineInfo("mac15-14_apple-m3-ultra_96gb_macos26", "Apple M3 Ultra · 96 GB",
                       "Apple M3 Ultra", "Mac15,14", 96, "26")


def test_profile_from_carries_fit_machine_and_meter():
    p = profile_from(_fit(), _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0)
    assert p.machine_id == "mac15-14_apple-m3-ultra_96gb_macos26"
    assert p.coefficients == {"a": 1.83, "b": 0.4}
    assert p.tier == "plug-calibrated (±4%)"
    assert p.meter["tier"] == "smart_plug"
    assert p.model_calibrated_on == "qwen3.6-27b"


def test_save_then_load_round_trips(tmp_path):
    p = profile_from(_fit(), _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0)
    path = save(p, root=str(tmp_path))
    assert os.path.isfile(path) and path.endswith("mac15-14_apple-m3-ultra_96gb_macos26.json")
    got = load(p.machine_id, root=str(tmp_path))
    assert got == p                                   # frozen dataclass equality
    assert [x.machine_id for x in list_profiles(root=str(tmp_path))] == [p.machine_id]


def test_load_missing_machine_returns_none(tmp_path):
    assert load("no-such-machine", root=str(tmp_path)) is None
