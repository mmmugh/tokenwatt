# src/tokenwatt/calibrator.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from tokenwatt import profiles
from tokenwatt.calibration import cal_scalar, certified
from tokenwatt.profiles import Profile

logger = logging.getLogger("tokenwatt.calibrator")


@dataclass(frozen=True)
class Applied:
    """The result of applying a calibration profile to one request's measured energy."""
    wall_j: float        # marginal WALL energy (J), predicted from the marginal rail energy
    tier: str            # honest tier label carried onto the ledger row, e.g. "plug-calibrated (±2.7%)"
    band_pct: float      # the profile's MEASURED confidence band


class Calibrator:
    """Runtime application of calibration profiles (C3).

    Resolves the per-(machine, model) profile once per model (cached for the life of the
    process) and converts a request's measured marginal RAIL energy to marginal WALL energy
    via the fitted E_wall ≈ a·rail_J + b·Δt. Only CERTIFIED (plug-calibrated) profiles are
    applied; an uncertified fit ("no tighter than estimated") is treated as no calibration,
    so a reported cost is never dressed up as calibrated when it isn't.

    Profiles are read at first use of a model and cached — recalibrating a model takes effect
    on the next `serve` (calibration is an offline step), not mid-process."""

    def __init__(self, machine_id: str, *, root: str | None = None) -> None:
        self._machine_id = machine_id
        self._root = root
        self._cache: dict[str, Profile | None] = {}

    def profile_for(self, model: str) -> Profile | None:
        if model not in self._cache:
            try:
                p = profiles.load(self._machine_id, model, root=self._root)
                if p is not None and not certified(p.meter.get("tier", ""), p.band_pct):
                    p = None                      # uncertified fit -> stay on the estimated tier
            except Exception as e:
                # A profile that became unreadable after startup (e.g. a truncated file from an
                # interrupted `calibrate fit`) must NOT break metering — price it as estimated,
                # and cache the miss so we log once, not on every request for this model.
                logger.warning("calibration profile unreadable for %r (%s); pricing as estimated",
                               model, type(e).__name__)
                p = None
            self._cache[model] = p
        return self._cache[model]

    def apply(self, model: str, marginal_rail_j: float, dt_s: float) -> Applied | None:
        p = self.profile_for(model)
        if p is None:
            return None
        wall_j = cal_scalar(p.coefficients["a"], p.coefficients["b"], marginal_rail_j, dt_s)
        return Applied(wall_j=wall_j, tier=p.tier, band_pct=p.band_pct)

    def available_models(self) -> list[str]:
        """Models this machine has a CERTIFIED profile for (used only for the serve banner)."""
        return [
            p.model_calibrated_on
            for p in profiles.list_profiles(root=self._root)
            if p.machine_id == self._machine_id and certified(p.meter.get("tier", ""), p.band_pct)
        ]
