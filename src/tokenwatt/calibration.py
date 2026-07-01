# src/tokenwatt/calibration.py
from __future__ import annotations

import math
from dataclasses import dataclass

from tokenwatt.nnls import nnls

_ESTIMATED_FLOOR_PCT = 15.0   # a plug fit no tighter than this isn't worth a distinct tier


def cal_scalar(a: float, b: float, e_rail_total_j: float, dt_s: float) -> float:
    """Pure prediction of marginal wall energy (J) from the scalar profile.
    Used at runtime (C3) — no solver, just arithmetic."""
    return a * e_rail_total_j + b * dt_s


def _rail_total(sample: dict) -> float:
    return sum(sample["e_rail_marginal_j"].values())


def fit_scalar(samples: list[dict]) -> tuple[float, float, float]:
    """Fit E_wall_marg ≈ a·E_rail_total + b·Δt (a,b ≥ 0). Returns (a, b, residual_rel)
    where residual_rel is the RMS relative residual over the samples."""
    A = [[_rail_total(s), s["dt_s"]] for s in samples]
    b_vec = [s["e_wall_marginal_j"] for s in samples]
    (a, b), _ = nnls(A, b_vec)
    rels = []
    for s in samples:
        pred = cal_scalar(a, b, _rail_total(s), s["dt_s"])
        actual = s["e_wall_marginal_j"]
        if actual > 0:
            rels.append(((pred - actual) / actual) ** 2)
    residual_rel = math.sqrt(sum(rels) / len(rels)) if rels else 0.0
    return a, b, residual_rel


def run_variance_rel(samples: list[dict]) -> float:
    """RMS across cells of each cell's pass-to-pass relative spread of marginal wall
    energy — the repeatability term. 0.0 when no cell has ≥2 passes."""
    by_cell: dict[str, list[float]] = {}
    for s in samples:
        by_cell.setdefault(s["cell"], []).append(s["e_wall_marginal_j"])
    spreads = []
    for vals in by_cell.values():
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            if mean > 0:
                spreads.append(((max(vals) - min(vals)) / mean) ** 2)
    return math.sqrt(sum(spreads) / len(spreads)) if spreads else 0.0


def confidence_band_pct(residual_rel: float, run_var_rel: float, meter_accuracy_pct: float) -> float:
    return 100.0 * math.sqrt(residual_rel ** 2 + run_var_rel ** 2 + (meter_accuracy_pct / 100.0) ** 2)


def tier_label(meter_tier: str, band_pct: float) -> str:
    if meter_tier == "smart_plug" and band_pct < _ESTIMATED_FLOOR_PCT:
        return f"plug-calibrated (±{band_pct:.1f}%)"
    return f"uncertified (±{band_pct:.1f}%) — no tighter than estimated"


@dataclass(frozen=True)
class FitResult:
    fit_type: str
    a: float
    b: float
    residual_rel: float
    run_variance_rel: float
    band_pct: float
    tier: str
    n_samples: int
    n_passes: int


def fit(campaign: dict) -> FitResult:
    samples = campaign["samples"]
    a, b, residual_rel = fit_scalar(samples)
    rv = run_variance_rel(samples)
    acc = campaign["meter"]["accuracy_pct"]
    band = confidence_band_pct(residual_rel, rv, acc)
    return FitResult(
        fit_type="scalar", a=a, b=b, residual_rel=residual_rel, run_variance_rel=rv,
        band_pct=band, tier=tier_label(campaign["meter"]["tier"], band),
        n_samples=len(samples), n_passes=campaign.get("passes", 1))
