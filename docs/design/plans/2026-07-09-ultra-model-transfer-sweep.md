# Ultra Model-Transfer Sweep — Plan (not yet executed)

**Status:** PLANNED 2026-07-09, decisions locked (3 passes; re-run 27b for a matched reference). Do not execute without a GO. Ultra-only; all models already local (no downloads).

## Goal
Extend the machine-transfer 1v1 finding along the **model axis**: does the scalar calibration
`E_wall = a·E_rail + b·Δt` — especially **b** — vary across model size/architecture on the *same*
machine (M3 Ultra)? We already know on the Ultra: `a` is nearly model-invariant (Qwen3.5-4B a=1.551
vs qwen3.6-27b a=1.547, +0.3%), but `b` differed ~11% (15.81 vs 17.68 W). The sweep spans a wider
range (dense→MoE, 4B→120B) to decide whether per-**model** keying earns its keep or one model's
calibration per machine generalizes.

## Models (all local, verified present)
| Slot | Model | serve target | size | status |
|---|---|---|---|---|
| small dense | Qwen3.5-4B-4bit | (done) | 2.5 GB | ✅ a=1.551 b=15.81 ±2.6% — pure reference (existing 5-dur fit; a,b are duration-independent so it's comparable) |
| **mid dense** | qwen3.6-27b-8bit | HF cache snapshot | 27 GB | ⬜ **RE-RUN** at 120/360/720×3 for a clean matched reference (the ±2.7% vardur fit used different durations) |
| **mid MoE** | `~/mlx-models/qwen3.6-35b-a3b-patched` | local path | 35 GB | ⬜ NEW |
| **large MoE** | `mlx-community/Qwen3-Coder-Next-6bit` | HF cache | 60 GB | ⬜ NEW |
| **huge MoE** | `mlx-community/gpt-oss-120b-MXFP4-Q8` | HF cache | 59 GB | ⬜ NEW |

Sweep RUNS **4 models**: qwen3.6-27b (re-run) + the 3 new MoEs. Qwen3.5-4B is the only pure existing reference.

## Protocol (per model — identical, for comparability)
- **3 durations: 120 / 360 / 720s** (6× Δt spread → pins `b`), **3 passes** (locked), standard 3 cells → **27 samples/model**.
- `--idle-seconds 900`, `--meter-host <studio-plug>` (Studio's plug — confirm it's still on .104 first).
- Fit each with the new **`calibrate fit d120.json d360.json d720.json`** (uses `fit_combined`).
- Compare (a,b) across all 5 models. Reference (a,b) come from the existing 5-duration fits.

## Sequencing (autonomous, one model at a time)
Memory: 96 GB total. gpt-oss-120b (59 GB) / Coder-Next (60 GB) won't co-exist with the mlx-tui slots
(43.8 GB). So, like the machine-transfer run:
1. **Stop the 3 mlx-tui slots** (`launchctl bootout user/501/com.stewie.{mlx-primary,mlx-vision,mlx-embed}`) to free ~44 GB. **Leave openclaw up** (gui/501, fragile). Trap-restore all 3 on exit/failure.
2. For each NEW model, sequentially:
   a. Launch `~/mlx-env/bin/python -m mlx_lm.server --model <target> --host 127.0.0.1 --port 8770`; wait for `/v1/models`; **read the served model-id** and use it verbatim as the campaign `--model`.
   b. Run the 3-duration campaign (resumable driver, 5×60s retry, `--out ~/.../data-sweep/<model>-d<dur>.json`).
   c. **Kill the server** (free memory) before the next model.
3. Restore the 3 mlx-tui slots (trap). openclaw untouched.
- Launch as a **direct background process on the Studio** (no Local-Network issue here — it's stewie's machine, permission long-granted; the Air's foreground-SSH constraint does NOT apply). Go hands-off during campaigns (idle-baseline contamination).

## Time
- 27 samples/model = 3×900 idle + 3 cells × 3 passes × (120+360+720) = 2700 + 10800 = **~3.75 h/model**.
- **4 models → ~15 h.** Too long for one night — split across **two nights** (e.g. night 1: 27b + 35B-A3B; night 2: Coder-Next + gpt-oss-120b), or one ~15 h window if available. Resumable, so either works.

## Verify-at-start (before committing the night)
1. **gpt-oss-120b serves under mlx_lm 0.31.3** (MXFP4 support) — do a 1-request smoke on port 8770 first; if it fails, that model drops out (or upgrade mlx_lm in a scratch env).
2. **Studio still metered by .104** (it was unplugged once — correlation-check: idle vs a short inference burst, like we did before).
3. **Memory headroom** with mlx-tui stopped: each big model + system < 96 GB (should be ~59+10 GB, fine).
4. **Model-id: pass the exact `--model` target as the campaign `--model`—do NOT read it from `/v1/models`.**
   mlx_lm.server's `/v1/models` lists ALL cached models (not the loaded one), and it serves the model
   named in each *request* (loading on demand), so reading `data[0]` there silently drives the wrong
   model. Warm the target up (a 1-token request) before the campaign so the on-demand load happens
   outside the idle baseline. (This bit night 1's first launch—caught by an id sanity-check in the log
   and fixed mid-run; see Results.)

## Deliverable
A 5-model table (a, b, band, wall/rail) + the verdict: is `b` model-dependent enough to require
per-model keying, or does per-machine (one model) generalize? Plus per-model linearity (the Ultra held
`a` flat across durations for 4B/27B; check the MoEs).

## Decisions (locked 2026-07-09)
- **3 passes** — chosen (tighter band).
- **Re-run qwen3.6-27b** at the matched 120/360/720 protocol — YES (adds it to the run list → 4 models).
- 6th fast-token model (Gemma-4-E4B, license permitting) — deferred; not in this sweep.

Remaining before a GO is just the run-time **verify-at-start** gates above (gpt-oss serves, .104 metering, memory, model-id).

## Results — complete 5-model sweep (M3 Ultra, plug-calibrated)

Night 1 (2026-07-14): 27b re-run + 35b-a3b. Night 2 (2026-07-15): Coder-Next + gpt-oss. Each sweep fit
is `calibrate fit d120 d360 d720` (fit_combined), 27 samples; all 12 sweep campaigns (both nights) passed
first-try. The 4B row is the earlier 5-duration 1v1 reference. Quant is the config-authoritative value now
recorded in every power-run record + profile (model names undersell it—see `read_quantization`).

| Model | Type | ~GB | Quant | `a` (rail→wall) | `b` (W per load-second) | Band |
|---|---|---|---|---|---|---|
| Qwen3.5-4B-4bit | small dense | 2.5 | 4-bit affine | 1.551 | 15.81 | ±2.6% |
| qwen3.6-27b-8bit | mid dense | 27 | 8-bit affine | 1.572 | 17.82 | ±2.8% |
| qwen3.6-35b-a3b (thinking) | mid MoE | 35 | 8-bit mixed | 1.561 | 20.62 | ±3.7% |
| Qwen3-Coder-Next-6bit | large MoE | 60 | 6-bit mixed | 1.506 | 21.61 | ±4.5% |
| gpt-oss-120b-MXFP4-Q8 | huge MoE | 59 | 4-bit mxfp4 mixed | 1.476 | 21.15 | ±2.7% |

Note `a` does not track quant simply: gpt-oss and 4B are both 4-bit yet have the most-different `a`
(1.476 vs 1.551), so the `a` drift is about size / memory-boundness, not bit-width.

**Verdict (full):**
- **`b` grows with memory footprint, then SATURATES.** 15.8→17.8→20.6, then a plateau at ~21–22 W for
  everything ≥35 GB. So `b ≈ f(model_GB)` holds only as a *saturating* curve—there is a ceiling to the
  per-second overhead (fans, regulators, memory subsystem max out), not unbounded growth.
- **`a` is NOT the clean machine constant night 1 implied (walk-back).** It looked flat across 4B→35B
  (1.55–1.57, ±0.7%), but the two huge MoEs pull it DOWN ~6% (gpt-oss 1.476, Coder-Next 1.506). gpt-oss'
  ±2.7% band puts that below the mid cluster for real; Coder-Next's ±4.5% is wider (suggestive). Likely
  cause: memory-bound huge MoEs dump more energy into the *measured* DRAM rail, so less rail→wall
  multiplier is needed. So `a` is roughly constant within a size class but drifts for huge models.
- **Per-(machine,model) keying is justified—more than night 1 said.** `a` and `b` partly cancel
  (higher-`a` models have lower `b`), so borrowing one model's profile for another costs ~2% on
  decode-heavy loads but up to ~5% on prefill-heavy loads at the 4B-vs-gpt-oss extreme—larger than the
  plug-calibrated bands. Matters at the plug tier; negligible for the estimated ±15–30% tier.
- **Reproducibility:** the 27b re-run (a=1.572, b=17.82) reproduced the earlier varying-duration 27b fit
  (a=1.547, b=17.68) within ~1.6% / ~0.8% across separate nights and durations.

Ops notes: linearity held (`a` flat across durations within each model). The 35b-a3b profile keyed on
its local *path* (served id), so it will not match a production model-id—re-key if used live. The bug
that bit night 1's first launch (reading the served id from mlx_lm.server `/v1/models`) is fixed in both
orchestrators; see verify-at-start item 4.
