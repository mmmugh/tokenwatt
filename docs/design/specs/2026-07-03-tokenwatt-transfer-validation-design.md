# TokenWatt — Model & Machine Transfer Validation + Battery-Aware Laptop Calibration Design Spec

**Status (2026-07-03):** M2 C0–C2 shipped. The M3 Ultra profile was just re-promoted from the
combined varying-duration fit (a=1.5467, b=17.68 W, band ±2.7%, n=18 over 5 durations, model
`qwen3.6-27b`). This phase decides *how far that one profile is allowed to travel* — across models
and across machines — and makes calibration honest on battery-powered laptops.

**This supersedes the old C5 "per-rail gated" plan.** The 2026-07-03 combined data showed
`corr(GPU_J, DRAM_J) = 0.998` across all inference workloads and the rail energy *mix* was ~constant
(~57% GPU / 40% DRAM) even between compute-bound prefill and bandwidth-bound decode. A per-rail fit
barely moved the residual (1.32% vs scalar 1.51%) and produced non-identifiable coefficients
(`a_cpu`=4.4). Per-rail cannot be estimated from inference load on this hardware; the condition-number
gate would reject it. So the transfer problem is answered by **per-model calibration**, not per-rail.

## Defaults taken (flag any to change at review)

- **Battery handling = gate, not subtract.** A laptop cell is used only if mean |battery power| during
  it is below a threshold (default 0.5 W). Subtracting would require modeling the adapter's
  charging-path efficiency; gating is honest and needs no model.
- **Light per-model campaigns.** b's functional form is already established, so each new model uses
  **2 durations × 2 passes** (default 180 s + 450 s) — enough for (a, b) + a real repeatability band
  in ~30–40 min, not the full 5-duration sweep.
- **Profile key = (machine_id, model).** Filenames become `<machine_id>__<model_slug>.json`; schema
  bumps to 2. Legacy schema-1 `<machine_id>.json` files still load, treated as the profile for their
  recorded `model_calibrated_on`.
- **Model-gated band.** The tight plug-calibrated band is claimed ONLY on an exact (machine, model)
  match. Any other model → `estimated (±15%)`, or that model's own profile if it exists.
- **Model list (verified current 2026-07-03).** Ultra sweep: qwen3.6-27b ✓, qwen3.6-35B-A3B (MoE),
  qwen3-coder-next (80B MoE), gpt-oss-120b. Shared Ultra+Air: Qwen3.5-4B, Gemma-4-E4B.

## 1. Problem & goal

The scalar profile `E_wall_marg ≈ a·E_rail_total + b·Δt` is fit on one model on one machine. `a` is a
weighted average over *that* model's GPU/DRAM/CPU energy mix; `b` is a machine-level constant-power
overhead (PSU/adapter/fans/board rails the SoC meter never sees). Two unknowns block trusting a
profile beyond where it was measured:

1. **Model transfer** — does a model with a very different compute/memory character (large MoE,
   4-bit edge model) land on the same (a, b), or does each model need its own?
2. **Machine transfer** — does a per-model profile hold on a *different* machine (scale, thermal,
   battery), or is machine-keying load-bearing? (We assume machine-keying is required; validate it.)

Plus a capability gap: on a laptop the wall plug reads machine + battery charging (or reads low under
adapter-exceeding load), so laptop users currently can't get an honest plug-calibrated profile.

**Goal:** measure the model- and machine-transfer spread with real data, decide the profile-keying
policy from it, and ship a battery-flux primitive that makes laptop calibration (and long-term
drift-checks) honest.

## 2. Non-goals

- **Per-rail (C5)** — discredited by the collinearity evidence above; dropped.
- **Model download/serving orchestration** — the user stages models on the inference backend manually.
- **The interactive wizard (C4)** — separate; this phase is measurement + the keying it implies.
- **Correcting normal per-request metering for battery** — unnecessary: normal metering reads the
  *rails*, which measure SoC draw regardless of power source. Battery only confounds the *wall
  reference* (calibration + drift-checks), which is the only place the new reader is used.

## 3. Final state — what this phase produces

- A `powersource.py` reader: sudoless instantaneous battery power + charge state; `None` on desktops.
- The campaign records a per-cell `battery_clean` flag; contaminated cells are flagged and excluded.
- `profiles.py` keyed by (machine_id, model) with schema 2 + legacy load; runtime lookup matches both.
- The second Shelly characterized (cadence, accuracy) and assigned to the Air.
- A dataset: (a, b, band) per model on the Ultra, and per shared-model on both Ultra and Air.
- A written verdict: **per-machine scalar suffices** vs **per-(machine, model) required**, with the
  measured spread that sets the model-transfer band, and a validated laptop calibration path.

## 4. Architecture

One new leaf module (`powersource.py`), a small integration into the existing campaign wall sampler,
an additive schema/lookup change in `profiles.py`, and a measurement protocol that reuses the shipped
campaign tooling. No change to the byte-exact proxy path.

## 5. Component A — `powersource.py` (battery-flux reader)

- `read_battery_flux() -> BatteryFlux | None`. Reads `ioreg -rn AppleSmartBattery` (sudoless),
  parses `InstantAmperage`, `Voltage`, `IsCharging`. Returns `BatteryFlux(net_w: float,
  charging: bool)` where `net_w` is signed instantaneous battery power (convention verified on the
  Air first run: `> 0` = battery absorbing/charging → wall over-reads machine; `< 0` =
  discharging/supplying → wall under-reads). Returns `None` when there is no real battery
  (`Voltage == 0` / device absent) — i.e. desktops like the Studio.
- Platform-gated with the rest (Apple Silicon). Pure-stdlib parse of the `ioreg` text; a
  `FakeBatterySource` test double returns scripted `(amperage, voltage, charging)` triples.
- **Sign-convention caveat:** Apple's `InstantAmperage` sign is verified empirically on the Air
  before any Air cell is trusted; the reader logs a one-line self-check on first use.

## 6. Component B — campaign battery gate

- `measure_idle` and `run_cell` sample `read_battery_flux()` alongside each wall reading. Each cell
  records `battery_net_w_mean` and `battery_clean = reader is None or mean|net_w| < GATE_W`
  (default 0.5 W).
- A cell that is not `battery_clean` is marked and **excluded from the fit**, with a logged reason and
  a wizard-facing hint ("charge to 100% / cap charging, then re-run"). Desktop: reader `None` →
  always clean → the Ultra path is byte-for-byte unchanged.
- The campaign JSON gains `battery_net_w_mean` + `battery_clean` per sample (additive; older readers
  ignore it).

## 7. Component C — per-(machine, model) profile keying + model-gated band

- Profile filename → `<machine_id>__<model_slug>.json` (`model_slug` = lowercased, `/`→`-`).
  `schema_version = 2`.
- `load(machine_id, model)` tries the per-model path first; falls back to a legacy
  `<machine_id>.json` **only if** its `model_calibrated_on` equals `model` (then optionally migrates
  it to the new name). `list_profiles` returns all (machine, model) profiles.
- Runtime (consumed by C3): `active_for(machine_id, model)` → exact match or `None`. On match →
  apply `cal_scalar` and stamp the profile's tier/band. On miss → identity + `estimated`; NEVER stamp
  `plug-calibrated` for an unmeasured model. `fail-loud` on unknown `schema_version`.

## 8. Component D — second-plug characterization

- Confirm it is a *metering* Shelly (Plus/Pro power-metering, not a non-metering unit). Run the C0
  `probe` against it to record accuracy_pct + accumulator cadence + `device_info()` identity.
- Assign it to the Air; each machine's profile records its own plug identity in the `meter` block
  (already supported). Machines are calibrated sequentially or in parallel (two plugs, two backends).

## 9. Experimental protocol

**Ultra — model-transfer sweep** (each: light 2×2 campaign, Core-4 battery, idle-subtracted, its
dedicated plug, OpenClaw paused as usual):

| model | type | ~4-bit size | fits |
|---|---|---|---|
| qwen3.6-27b | dense 27B | ~15 GB | Ultra ✓ (done) |
| qwen3.6-35B-A3B | MoE, 3B active | ~20 GB | Ultra |
| qwen3-coder-next | MoE 80B, 3B active | ~45 GB | Ultra |
| gpt-oss-120b | MoE 120B | ~63 GB | Ultra |

**Cross-machine — shared small models on BOTH Ultra and Air (16 GB, battery-gated):**

| model | type | ~4-bit size |
|---|---|---|
| Qwen3.5-4B | dense 4B | ~2.5 GB |
| Gemma-4-E4B | edge ~4.5B | ~2.5–3 GB |

Before each Air run, verify residency (no swap) via `lms ps`/memory check — swap would pollute the
power signal.

## 10. Analysis & decision criteria

- **Model transfer holds** if each model's (a, b) predicts the *other* models' wall energy within the
  measured band (equivalently, the per-model `a` values cluster within ~±band with `b` ~machine-
  constant). → per-machine scalar suffices; keep the keying infra but a machine may share one profile.
- **Model transfer fails** if the spread exceeds the band. → per-(machine, model) keying is required;
  the observed spread sets the honest band for applying a profile to an *un-calibrated* model.
- **Machine transfer:** compare the shared models' (a, b) on Ultra vs Air. Expectation: `b` differs
  (adapter vs PSU, fanless throttling); confirms machine-keying is load-bearing.
- **Laptop method validated** if the Air's cells pass the battery gate and its band lands under the
  15% floor.

## 11. Error handling & failure modes

- Battery never quiet on the Air → cells fail the gate; surface the 100%/AlDente guidance; do not emit
  a profile from contaminated cells (fail loud, no fake band).
- Model swaps to disk on the Air → residency check aborts before the run.
- Second plug turns out non-metering → stop; a non-metering plug cannot calibrate (honesty contract).
- Unknown `schema_version` at load → raise, don't guess.

## 12. Testing (intent, not just coverage)

- `powersource`: FakeBatterySource → charging (net_w>0), discharging (net_w<0), no-battery (None);
  assert the gate flags contaminated and passes clean; assert desktop path is unchanged.
- keying: (machine, model) match returns the profile; a *different* model returns `None` (→ estimated)
  — the test must fail if a mismatched model ever gets a plug-calibrated band.
- legacy load: a schema-1 file loads iff its `model_calibrated_on` matches the requested model.
- sign convention: a regression test pins the documented sign once verified on the Air.

## 13. Open questions (resolve during execution, do not block)

- Battery sign convention — verify on the Air's first run; then pin it in a test.
- Gate threshold (0.5 W default) — tune on the Air.
- Exact quant/context per model to guarantee residency (esp. gpt-oss-120b on 96 GB, 4B models on 16 GB).
- Whether to auto-migrate the legacy Ultra profile to the `__model` name on first schema-2 write.

## 14. Staged plan

1. **A — `powersource.py`** + tests (mockable, desktop no-op). Verifiable now, no hardware needed.
2. **B — campaign battery gate** + tests; confirm Ultra path byte-unchanged (desktop reader = None).
3. **C — per-(machine, model) keying** + legacy load + model-gated lookup + tests.
4. **D — second-plug characterization** (one-off measurement).
5. **E — run the sweep**: Ultra models, then shared models on both machines; light 2×2 each.
6. **Analysis & verdict** → decide keying policy; write it up; feed C3 (runtime apply) the lookup.
