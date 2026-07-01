# src/tokenwatt/profiles.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo

_PROFILE_SCHEMA = 1
_DEFAULT_ROOT = "~/.tokenwatt/profiles"


def _root(root: str | None) -> str:
    return os.path.expanduser(root if root is not None else _DEFAULT_ROOT)


@dataclass(frozen=True)
class Profile:
    machine_id: str
    label: str
    fit_type: str
    coefficients: dict
    residual_rel: float
    run_variance_rel: float
    band_pct: float
    tier: str
    meter: dict                      # {name, tier, accuracy_pct, model?, gen?, mac?}
    n_samples: int
    n_passes: int
    model_calibrated_on: str
    macos: str
    created_at: float
    schema_version: int = _PROFILE_SCHEMA


def profile_from(fit: FitResult, machine: MachineInfo, meter: dict, *,
                 model_calibrated_on: str, created_at: float) -> Profile:
    return Profile(
        machine_id=machine.machine_id, label=machine.label, fit_type=fit.fit_type,
        coefficients={"a": fit.a, "b": fit.b}, residual_rel=fit.residual_rel,
        run_variance_rel=fit.run_variance_rel, band_pct=fit.band_pct, tier=fit.tier,
        meter=meter, n_samples=fit.n_samples, n_passes=fit.n_passes,
        model_calibrated_on=model_calibrated_on, macos=machine.macos_major,
        created_at=created_at)


def save(profile: Profile, *, root: str | None = None) -> str:
    d = _root(root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{profile.machine_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return path


def load(machine_id: str, *, root: str | None = None) -> Profile | None:
    path = os.path.join(_root(root), f"{machine_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return Profile(**json.load(f))


def active_for_current_machine(machine_id: str, *, root: str | None = None) -> Profile | None:
    return load(machine_id, root=root)


def list_profiles(*, root: str | None = None) -> list[Profile]:
    d = _root(root)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as f:
                out.append(Profile(**json.load(f)))
    return out
