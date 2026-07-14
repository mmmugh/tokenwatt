# src/tokenwatt/calibration.py
from __future__ import annotations

import math
from dataclasses import dataclass

from tokenwatt.nnls import nnls

_ESTIMATED_FLOOR_PCT = 15.0   # a plug fit no tighter than this isn't worth a distinct tier
GATE_W = 0.5     # a laptop cell is trusted only if |battery power| stayed below this (calibration only)


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
    n_excluded: int = 0


def _contaminated(sample: dict) -> bool:
    b = sample.get("battery_abs_w")
    return b is not None and b >= GATE_W


def _spans_multiple_durations(samples: list[dict]) -> bool:
    """True if any cell name spans clearly-different Δt — the fingerprint of a
    hand-merged multi-duration campaign. A single campaign runs every cell at one
    fixed duration, so >10% spread of dt_s within a cell means samples from different
    durations were pooled, which `fit()` cannot band honestly (see `fit_combined`)."""
    by_cell: dict[str, list[float]] = {}
    for s in samples:
        by_cell.setdefault(s["cell"], []).append(s["dt_s"])
    return any(min(dts) > 0 and (max(dts) - min(dts)) / min(dts) > 0.10
               for dts in by_cell.values())


def fit(campaign: dict) -> FitResult:
    idle_batt = (campaign.get("idle") or {}).get("battery_abs_w")
    if idle_batt is not None and idle_batt >= GATE_W:
        raise ValueError(
            f"idle baseline battery-contaminated ({idle_batt:.2f} W ≥ {GATE_W} W): the marginal "
            f"subtraction is unreliable — charge to 100% / cap charging and re-run")
    all_samples = campaign["samples"]
    samples = [s for s in all_samples if not _contaminated(s)]
    n_excluded = len(all_samples) - len(samples)
    if not samples:
        raise ValueError(
            "no clean samples to fit: every cell was battery-contaminated — "
            "charge to 100% / cap charging and re-run")
    if _spans_multiple_durations(samples):
        raise ValueError(
            "fit() received samples spanning multiple durations within a cell (a "
            "hand-merged multi-duration campaign): its pass-repeatability groups by "
            "cell name across durations and would inflate the band — use fit_combined() "
            "with one campaign per duration instead")
    a, b, residual_rel = fit_scalar(samples)
    rv = run_variance_rel(samples)
    acc = campaign["meter"]["accuracy_pct"]
    band = confidence_band_pct(residual_rel, rv, acc)
    return FitResult(
        fit_type="scalar", a=a, b=b, residual_rel=residual_rel, run_variance_rel=rv,
        band_pct=band, tier=tier_label(campaign["meter"]["tier"], band),
        n_samples=len(samples), n_passes=campaign.get("passes", 1), n_excluded=n_excluded)


def fit_combined(campaigns: list[dict]) -> FitResult:
    """Fit E_wall_marg ≈ a·E_rail + b·Δt across MULTIPLE single-duration campaigns.
    Varying Δt across campaigns is what pins the per-time term `b`. `a`,`b` and the
    fit residual come from the pooled samples; pass-to-pass repeatability is measured
    WITHIN each campaign (one duration) and RMS-combined — never across durations.

    Use this instead of `fit()` on a hand-merged campaign: `fit()` groups repeatability
    by cell NAME, so a merged 'prefill' spanning 120s→720s makes (max−min)/mean explode
    and the band balloons (e.g. ±150%) even when each duration is tightly repeatable."""
    if not campaigns:
        raise ValueError("fit_combined needs at least one campaign")
    for c in campaigns:
        idle_batt = (c.get("idle") or {}).get("battery_abs_w")
        if idle_batt is not None and idle_batt >= GATE_W:
            raise ValueError(
                f"idle baseline battery-contaminated ({idle_batt:.2f} W ≥ {GATE_W} W): the marginal "
                f"subtraction is unreliable — charge to 100% / cap charging and re-run")
    clean_by_campaign = [[s for s in c["samples"] if not _contaminated(s)] for c in campaigns]
    pooled = [s for cs in clean_by_campaign for s in cs]
    n_excluded = sum(len(c["samples"]) for c in campaigns) - len(pooled)
    if not pooled:
        raise ValueError(
            "no clean samples to fit: every cell was battery-contaminated — "
            "charge to 100% / cap charging and re-run")
    a, b, residual_rel = fit_scalar(pooled)
    per_campaign_rv = [run_variance_rel(cs) for cs in clean_by_campaign if cs]
    rv = math.sqrt(sum(x * x for x in per_campaign_rv) / len(per_campaign_rv)) if per_campaign_rv else 0.0
    acc = campaigns[0]["meter"]["accuracy_pct"]
    band = confidence_band_pct(residual_rel, rv, acc)
    return FitResult(
        fit_type="scalar-combined", a=a, b=b, residual_rel=residual_rel, run_variance_rel=rv,
        band_pct=band, tier=tier_label(campaigns[0]["meter"]["tier"], band),
        n_samples=len(pooled), n_passes=campaigns[0].get("passes", 1), n_excluded=n_excluded)
