# Changelog

All notable changes to TokenWatt. This project adheres to [Semantic Versioning](https://semver.org).

## [0.3.0] - 2026-07-16

The self-calibration release: measure your machine's true energy accuracy against a wall meter and
have the proxy price from it.

### Added
- **Self-calibration loop** (`tokenwatt calibrate probe|campaign|fit`). A Shelly Gen2+ smart plug
  provides wall-energy ground truth; a Core-4 text load campaign captures idle-subtracted marginal
  rail/wall samples; a pure-Python Lawson-Hanson NNLS fit produces a per-(machine, model) profile
  `E_wall ≈ a·rail_J + b·Δt` with a **measured** confidence band and an honest tier.
- **Runtime application of the calibration profile (C3).** `serve` loads this machine's *certified*
  profiles and prices matching requests from measured wall energy, labeling their ledger rows
  `plug-calibrated (±measured%)`. Costs now genuinely change from `estimated` to `calibrated` after
  you calibrate; an uncertified fit (no tighter than estimated) is never applied.
- **`tokenwatt doctor`:** one-shot diagnosis of config, proxy, upstreams, routing, ledger, and the
  energy meter. `--json` for bug reports, `--fix` for safe repairs.
- **Dynamic model→upstream discovery** (opt-in): routes to wherever a model is actually loaded, so a
  swapped mlx-tui slot or `lms load` needs no config edit. The proxy also answers `GET /v1/models`
  as an aggregated, unmetered list.
- **Transfer validation + battery awareness.** Profiles are keyed per-(machine, model); the campaign
  reads battery flux sudolessly, excludes battery-contaminated cells, and fails loud on a dirty idle
  baseline (`--idle-seconds` sizes the idle window independently). Each profile records the served
  model's quantization (bits / group_size / mode / mixed).
- Leak-scan git hooks (`.githooks/` pre-commit + pre-push) and `scripts/setup.sh` to activate them.

### Changed
- `report` and `wrap` are calibration-aware. `report` gains a per-model `conf` column (the measured
  band, or `est`) and its banner reflects whether calibration is active; `wrap` drops the
  "estimated ±15–30%" caveat once every metered model is plug-calibrated.
- Ledger gains `calibrated` and `calib_band_pct` columns (forward-only migration; existing ledgers
  upgrade automatically). `by_model()` surfaces per-model calibration status.
- CI runs the full supported Python range (3.10–3.14) on current GitHub Action majors.

### Deferred / not yet implemented
- **Time-of-use (TOU) / tiered rate pricing** is not implemented; v0.3.0 prices at a single flat
  `$/kWh` (`--rate`, or `rate.flat_usd_per_kwh`). The `tokenwatt rate` CLI, URDB lookup, and the
  `rate_period` ledger column from the original design spec are deferred.
- An interactive `calibrate` wizard and `calibrate show` (C4), and the per-rail fit upgrade (C5), are
  deferred; calibration ships as the three explicit subcommands.

## [0.2.0] - 2026-06-21

First public PyPI release: the OpenAI-compatible metering proxy (byte-exact passthrough, per-rail
IOReport energy bracketing, idle subtraction, flat-rate pricing, sqlite ledger) with
`serve` / `report` / `compare` / `wrap` / `init`, static model→upstream routing, and the dated cloud
price comparison.
