# src/tokenwatt/profiles.py
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo

_PROFILE_SCHEMA = 2
_DEFAULT_ROOT = "~/.tokenwatt/profiles"


def _root(root: str | None) -> str:
    return os.path.expanduser(root if root is not None else _DEFAULT_ROOT)


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


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
    path = os.path.join(d, f"{profile.machine_id}__{_model_slug(profile.model_calibrated_on)}.json")
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return path


def _read(path: str) -> Profile:
    with open(path) as f:
        data = json.load(f)
    sv = data.get("schema_version")
    if sv not in (1, 2):
        raise ValueError(f"unknown profile schema_version {sv!r} in {path}")
    return Profile(**data)


def load(machine_id: str, model: str, *, root: str | None = None) -> Profile | None:
    d = _root(root)
    keyed = os.path.join(d, f"{machine_id}__{_model_slug(model)}.json")
    if os.path.isfile(keyed):
        p = _read(keyed)
        if p.model_calibrated_on == model:
            return p
    legacy = os.path.join(d, f"{machine_id}.json")           # pre-schema-2 single-model file
    if os.path.isfile(legacy):
        p = _read(legacy)
        if p.model_calibrated_on == model:
            return p
    return None


def active_for(machine_id: str, model: str, *, root: str | None = None) -> Profile | None:
    return load(machine_id, model, root=root)


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
