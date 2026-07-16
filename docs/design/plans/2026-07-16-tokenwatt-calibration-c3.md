# C3 — Runtime application of the calibration profile

**Status:** IN PROGRESS 2026-07-16 (decided for v0.3.0). Closes the loop the README already
promises: costs read `estimated (±15-30%)` **until you calibrate**, then re-price and re-label to
the plug-calibrated tier. Before C3, `calibrate fit` wrote a profile that nothing loaded.

## What was missing (verified)

`proxy.py`/`cost.py` never referenced a profile; the ledger had no calibration columns; `profiles.py`
was imported only by the `calibrate` CLI. So calibrating changed nothing a user could see.

## Design

The runtime primitive already exists: `calibration.cal_scalar(a, b, e_rail_total_j, dt_s)` (documented
"Used at runtime (C3)"). C3 wires it in.

- **`calibration.certified(meter_tier, band_pct)`** — factored out of `tier_label` as the single source
  of truth: a profile is a real calibration iff a metering plug produced a band tighter than the
  estimated floor (15%). Drives both the label and the runtime gate.
- **`calibrator.Calibrator(machine_id, root=...)`** — resolves the per-(machine, model) profile once per
  model (cached), and `apply(model, marginal_rail_j, dt_s)` returns `Applied(wall_j, tier, band_pct)` or
  `None`. Only CERTIFIED profiles are applied; an uncertified fit ("no tighter than estimated") is
  treated as no calibration, so a cost is never dressed up as calibrated. Cached at first use —
  recalibrating takes effect on the next `serve` (calibration is an offline step).
- **`proxy._finalize`** — when a profile applies, `kwh = cal_scalar(a, b, marginal_j, dt)/3.6e6` instead
  of `marginal_j/3.6e6`; `energy_confidence` carries the tier; the row is flagged `calibrated`.
- **`ledger`** — forward-only migration adds `calibrated INTEGER DEFAULT 0` and `calib_band_pct REAL`;
  `by_model()` surfaces `n_calibrated` + a representative band.
- **`report` / `wrap`** — become tier-aware: the header and per-model `conf` column show the measured
  band for calibrated models and `est` otherwise; `wrap`'s "estimated, pre-calibration" caveat drops
  once every metered model is calibrated. `compare` makes no "estimated" claim, so it is unchanged.

## Explicitly deferred (out of scope for C3/v0.3.0)

- Cold-start (`model_loads`) energy stays raw rail J — it is a separate, separately-labeled row.
- Per-rail correction (C5) and an interactive `calibrate` wizard / `calibrate show` (C4).

## Honesty invariants (tested)

Uncertified profiles never re-price. A calibrated row's `energy_confidence` is the measured tier, never
a fixed "±2–5%". No fabricated `$0`. Existing estimated-tier behavior is unchanged when no profile loads.
