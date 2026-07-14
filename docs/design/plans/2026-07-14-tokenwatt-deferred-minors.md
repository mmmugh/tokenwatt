# TokenWatt Deferred Minors — Fix Plan (queued 2026-07-14)

Six small, independent, testable fixes flagged in the transfer-validation reviews. Each is
self-contained; do them TDD (failing test → fix → verify) — good candidate for
subagent-driven-development (one task per minor) or direct inline TDD. Line numbers are as of
0.2.56; re-confirm before editing. Keep the honesty contract and the "tests encode intent" rule.

## M1 — `fit()` guards against merged multi-duration data
- **Where:** `src/tokenwatt/calibration.py` — `fit()` (~L83); `fit_combined()` is at L106.
- **Problem:** `fit()` on a *hand-merged* campaign (samples spanning multiple durations) computes
  `run_variance_rel` by cell NAME across durations → band balloons (±150%). The CLI now routes
  multi-file through `fit_combined`, so the normal path is safe, but a direct `fit(merged)` call is
  silently wrong.
- **Fix:** in `fit()`, detect >1 distinct duration among samples (cluster `dt_s`, e.g. >~10% spread
  within a cell name) and raise a clear `ValueError` pointing at `fit_combined` — or auto-delegate.
- **Test:** `fit()` on a 2-duration merged campaign raises/handles instead of returning a garbage band.

## M2 — idle battery sampled *during* the window, not just boundaries
- **Where:** `campaign.py` — `measure_idle` (~L158): reads battery at start (L165) + end (L170) only,
  around `sleep(seconds)` (L166).
- **Problem:** a mid-idle charge burst isn't caught → the calibration gate can miss idle contamination.
- **Fix:** poll battery ~every `battery_poll_s` across the idle window (mirror `run_cell`'s loop at
  L197–199), folding into `batt_max` via `_batt_peak`.
- **Test:** `measure_idle` with a `FakeBatterySource` that spikes only mid-window → `batt_max` reflects
  the spike (today it wouldn't).

## M3 — `_model_slug` guards degenerate/unknown model
- **Where:** `profiles.py` — `_model_slug` (~L20), used at save (L57) + keyed lookup (L74).
- **Problem:** a campaign with `model == "?"` (unknown) yields a degenerate slug → `<machine>__.json`,
  polluting the keyed profiles.
- **Fix:** when model is `"?"`/empty, don't write a keyed profile (fall back to the legacy machine-only
  name) or raise. Save and lookup must agree.
- **Test:** `profile_from`/`save` with `model="?"` does not produce a degenerate `<machine>__.json`.

## M4 — `list_profiles` uses the fail-loud reader
- **Where:** `profiles.py` — `list_profiles` (L91) currently does `Profile(**json.load(f))` (L99);
  `_read` (L63) is the fail-loud reader (schema check).
- **Fix:** `list_profiles` should route each file through `_read` (consistent schema validation +
  fail-loud on unknown schema) instead of a bare `Profile(**)`.
- **Test:** `list_profiles` over a malformed/unknown-schema profile fails loud like `_read`, not a raw
  `TypeError`.

## M5 — persist `n_excluded` into the Profile
- **Where:** `profiles.py` — `Profile` (L25) + `profile_from` (L43). `FitResult.n_excluded` exists.
- **Problem:** the count of battery-contaminated cells excluded from the fit is console-only, not stored.
- **Fix:** add `n_excluded` to the `Profile` dataclass + `profile_from`; bump the profile schema if
  needed and keep `_read` back-compatible (default 0 for old profiles).
- **Test:** `profile_from` carries `n_excluded`; round-trips through `save` → `_read`.

## M6 — stop re-spawning `ioreg` on desktops
- **Where:** `campaign.py` — `run_cell` battery poll (L197–199, `battery_poll_s=5.0`); `_default_battery`
  = `IOKitBatterySource` always (L81–82).
- **Problem:** on a desktop (no battery), `IOKitBatterySource.read_flux` spawns `ioreg` every ~5 s/cell
  and always returns `None` — pure waste.
- **Fix:** once the battery source yields `None` (desktop), short-circuit and stop polling it for the
  rest of the run. (Cheapest: a "seen None → give up" latch in the poll path.)
- **Test:** `run_cell` with a `FakeBatterySource` whose first read is `None` calls `read_flux` once, not
  every poll (assert call count).

## Done-criteria
All six have a test that fails before the fix and passes after; full suite green; no behavior change to
the honesty contract or the normal (single-campaign) fit/serve paths.
