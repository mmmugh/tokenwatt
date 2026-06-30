# TokenWatt — Calibration Loop Design Spec

- **Status:** Draft for review
- **Date:** 2026-06-30
- **Author:** Justin Stewart (with Claude)
- **Scope:** The Shelly-driven self-calibration loop (spec §8, features #3 + #4). One coherent
  vertical: *smart-plug reads → systematic workload battery → fit → per-machine profile with a
  real confidence band → ledger rows that legitimately read `calibrated`.* The TOU rate model (#2)
  and machine-readable `--json`/`--once` output — also nominally "M2" — are **separate
  sub-projects** and out of scope here.

An interactive tool, `tokenwatt calibrate`, that guides a user to connect a metering smart plug
(Shelly Plus Plug US), runs a standardized inference workload battery across their configured
models under sustained load, and fits a **repeatable correlation** between TokenWatt's zeus
per-rail energy and the plug's measured AC wall energy. The output is a per-machine calibration
profile that turns the proxy's `estimated (±15–30%)` numbers into honestly-banded
`plug-calibrated` numbers — and is designed so the WT310E lab tier and a per-rail fit are
drop-in upgrades, not rewrites.

---

## Defaults taken (flag any to change at review)

| Decision | Default | Change to… |
|---|---|---|
| Fit ambition | **staged** — scalar floor now, per-rail gated on the data (C5) | commit to per-rail, or stay scalar |
| Wall meter | **Shelly Plus Plug US** (Gen2 RPC, `aenergy.total`) | add WT310E lab adapter (deferred) |
| Calibration driver | **self-contained harness** (owns its own meter, hits upstream directly) | drive load through the running proxy |
| Profile store | **one human-readable JSON per machine** under `~/.tokenwatt/profiles/` | a sqlite table |
| Profile key | **machine** (SoC + model id + RAM + macOS major) | per-model, or per-(machine×model) |
| Load source | **standardized prompts, user's configured models as the variable** | a fixed model-independent battery |
| Smart-plug tier | **distinct tier below the lab tier**, band derived not asserted | merge tiers |

---

## 1. Problem & goal

Today every TokenWatt cost renders `estimated (±15–30%)`: the honesty contract is wired in, but
`Cal` is implicitly the identity — there is no calibration code at all (no `calibrate` command, no
`MeterSource`, no profile registry, no `calib_*` ledger columns). zeus reports Apple's CLPC
**modeled** on-die estimate (Digital Power Estimator), which carries an unknown per-machine offset
and omits PSU AC/DC conversion loss. We have a metering smart plug; the goal is to **measure that
offset and remove it**, per machine, with a stated error band.

**Primary goal:** a guided, repeatable campaign that produces a per-machine profile mapping zeus
marginal rail energy → AC wall energy, such that the proxy can stamp ledger rows
`plug-calibrated (±x%)` where `x` is *derived from measured residual + run-to-run variance + meter
accuracy*, never asserted.

**Success criteria**
- `tokenwatt calibrate` walks a first-time user from "I have a Shelly" to an active profile with
  no prior knowledge of the math, and reports the achieved band in plain language.
- A profile is **machine-keyed**; the M3 Ultra and the M4 Air carry separate profiles and never
  cross-apply.
- The correlation is **repeatable**: ≥2 passes of the battery agree within the reported band, or
  the run widens the band / refuses to certify — never silently.
- The proxy, with a profile loaded, prices requests `plug-calibrated`; with none, `estimated`. A
  Shelly fit can **never** print the lab-grade `±2–5%`.
- The whole campaign→fit→profile→runtime path runs deterministically in CI on fake hardware.

## 2. Non-goals

- **TOU rate model (#2)** and **`--json`/`--once`** — separate M2 sub-projects.
- **Per-PID attribution** — impossible on Apple Silicon; we calibrate whole-SoC↔whole-wall.
- **Reproducing the WT310E lab campaign** — the lab adapter and `lab-calibrated` tier are
  *designed-for, not built* here.
- **Non-Apple-Silicon** — unchanged platform marker; tests use fakes.

## 3. Final state — what the loop produces

A per-machine **calibration profile**: a small JSON artifact storing the fit (scalar or gated
per-rail), its coefficients, the residual, the meter's tier + accuracy, sample/run counts, macOS
version, a timestamp, and the derived confidence band + tier label. When the proxy boots, it loads
the profile matching the current machine, applies `Cal` to each request's marginal rail energy to
get calibrated kWh, and stamps the row with `calib_profile_id` + `calib_confidence`. No match →
`Cal = identity`, `estimated`. This is the honesty contract made real instead of asserted.

## 4. Architecture — one self-contained calibration harness

`tokenwatt calibrate` is a **self-contained harness**, not a load generator pointed at the running
HTTP proxy. It owns the same primitives the proxy uses — `ZeusMeter` + `IdleBaseline` + the
marginal convention — plus a `MeterSource` (the Shelly) and an OpenAI client. It sends requests
straight to a configured route's `upstream`, brackets each window with the meter, and reads the
plug's accumulator at the same window boundaries.

**Why self-contained.** What the proxy applies `Cal` to at runtime is per-request *marginal rail
energy* measured by `ZeusMeter` under the marginal convention. If the harness measures rails with
the *same* primitive under the *same* convention, the fitted relationship transfers directly —
without coupling calibration to a separately-running server, and without folding HTTP-proxy
bracketing overhead into the fit. We calibrate the SoC↔wall relationship, which is what matters,
not the socket layer.

```
 ┌─────────────────────────── tokenwatt calibrate ───────────────────────────┐
 │  machine probe → MeterSource(Shelly) → pre-flight                          │
 │            │                                                               │
 │   ┌────────┴────────┐   for each (model × load-cell), sustained Δt:        │
 │   │ load driver     ├─► OpenAI request(s) to route.upstream                │
 │   │ (standardized   │                                                      │
 │   │  prompts)       │   bracket window:                                    │
 │   └────────┬────────┘     E_rail = ZeusMeter.end − idle_rail·Δt  (marginal)│
 │            │              E_wall = Shelly.Δwh   − idle_wall·Δt   (marginal)│
 │            ▼                                                               │
 │   sample table  [(E_rail[], E_wall, Δt, cell, model)]  × ≥2 passes         │
 │            │                                                               │
 │            ▼                                                               │
 │   fit (scalar floor; per-rail gated) → band → tier                         │
 │            │                                                               │
 │            ▼                                                               │
 │   ~/.tokenwatt/profiles/<machine_id>.json   ──(boot)──►  proxy applies Cal │
 └───────────────────────────────────────────────────────────────────────────┘
```

## 5. Component A — `MeterSource`

A minimal protocol for a wall-energy reference:

```python
class MeterSource(Protocol):
    name: str
    accuracy_pct: float           # the meter's own ± accuracy (drives the band floor)
    tier: str                     # "smart_plug" | "manual" | "lab"
    def read_accumulated_wh(self) -> float: ...   # monotonic Wh counter
```

We integrate by **reading the accumulator at window boundaries** (`Δwh = read(t1) − read(t0)`),
never by integrating instantaneous power ourselves — the plug's own counter is more accurate and
avoids sampling error.

- **`ShellyMeterSource`** (the only built implementation): Gen2 RPC over HTTP,
  `GET http://<host>/rpc/Switch.GetStatus?id=<id>` → `result.aenergy.total` (Wh, milli-Wh
  resolution). Configured by host/IP + switch id; optional digest auth if the plug has a password.
  `accuracy_pct ≈ 1`, `tier="smart_plug"`. Reachability + identity surfaced via a `doctor`-style
  check.
- **`ManualMeterSource`** (trivial fallback): prompts the user to type a Kill-A-Watt cumulative
  reading at each boundary. `tier="manual"`, wider accuracy.
- **`LabMeterSource` (WT310E)** — *designed-for, not built.* The interface must not preclude a
  serial/CSV accumulator ingest; the `lab` tier and `±2–5%` band are gated behind it.

**Resolution caveat (load-bearing):** a smart plug's accumulated counter updates on a coarse
internal cadence and has finite resolution. Windows must be **long relative to that cadence** for
the wall Δ to dwarf quantization noise. We do **not** assume the cadence — C0 measures it
empirically before any cell duration is fixed.

## 6. Component B — Workload battery + synchronized capture

A fixed set of **load cells** chosen so the rails vary *independently* across the battery (the
statistical precondition for ever earning the per-rail tier):

| Cell | Prompt / params | Stresses |
|---|---|---|
| `idle` | no inference; quiesced machine | baseline (rails + wall) |
| `prefill` | long prompt (~4k tok), `max_tokens` small | compute-bound (GPU) |
| `decode` | tiny prompt, large `max_tokens`, EOS suppressed | memory-bound (DRAM) |
| `balanced` | medium prompt, medium generation | mixed |
| `embeddings` | batched embed calls *(only if an embeddings route exists)* | embed path |
| `vision` | image + prompt *(only if a vision route exists)* | vision path |

Prompts are **standardized and shipped with the package** (deterministic, version-stamped into the
profile); the **model is the variable** — the campaign runs each applicable cell against each
user-selected route. Different models stressing rails differently is *desirable*: it adds
independent variation to the design matrix.

Each cell runs as **sustained load** — the driver loops requests (optionally with bounded
concurrency) for a fixed wall-clock duration `Δt` so the plug's accumulator delta is large vs its
resolution. For each cell run we record one sample:

```
sample = {
  cell, model, dt,
  e_rail_marginal: {cpu_total, gpu, gpu_sram, dram, ane},   # zeus active − idle_rail·dt
  e_wall_marginal: float (J),                               # plug Δwh→J − idle_wall·dt
  tok_in, tok_out,                                          # bookkeeping / sanity
}
```

`idle` is measured at **both** rails and wall, yielding `idle_rail` power per rail and
`idle_wall` power (W); these define the marginal subtraction applied to every other cell, so both
sides of the fit are marginal — matching the proxy's per-request `e_marginal_j`.

## 7. Component C — The fit (`calibration.py`)

Both sides marginal; energy in joules. Two models, one gate.

**Scalar floor (always identifiable):**
```
E_wall_marg  ≈  a · E_rail_total  +  b · Δt          # NNLS: a ≥ 0, b ≥ 0
```
`a` absorbs the DPE offset + PSU loss as a single slope; `b·Δt` captures any marginal,
time-proportional active overhead not carried by the rails (fan/PSU curvature). Robust under *any*
sustained load.

**Per-rail ceiling (gated, C5):**
```
E_wall_marg  ≈  β₀ · Δt  +  Σ_rail β_rail · E_rail   # NNLS, all β ≥ 0
```

**Promotion gate** — emit per-rail only when *all* hold, else fall back to scalar and say why:
1. enough independent samples (≥ a threshold count across cells × models × passes);
2. design-matrix condition number / per-rail VIF below threshold (the rails *actually* moved
   independently — otherwise coefficients are meaningless);
3. per-rail residual beats scalar by a margin on **held-out cells** (not in-sample);
4. all β ≥ 0 (NNLS guarantees this; a clamped coefficient is a smell the gate logs).

**Confidence band** = relative residual RMSE (held-out) ⊕ run-to-run variance (across the ≥2
passes) ⊕ the meter's `accuracy_pct`, combined in quadrature → a single `±%`. **Tier** is a
function of meter tier + fit type + band magnitude:

| Meter | Fit | Tier label |
|---|---|---|
| — (none) | identity | `estimated (±15–30%)` |
| smart_plug | scalar or gated per-rail | `plug-calibrated (±x%)` |
| lab (WT310E) | per-rail | `lab-calibrated (±2–5%)` *(deferred)* |

The band is **measured, not asserted**: non-repeatable passes widen it or block certification.

## 8. Component D — Profile registry (`profiles.py`)

One human-readable JSON file per machine under `~/.tokenwatt/profiles/<machine_id>.json` —
transparent, inspectable, shareable (same ethos as the dated, editable `cloud.py` price table).

- **`machine_id`** = SoC marketing name (`sysctl machdep.cpu.brand_string` → "Apple M3 Ultra") +
  model id (`sysctl hw.model`) + RAM (`hw.memsize`) + macOS major (`platform.mac_ver`), hashed/
  slugged to a stable key. No sudo (consistent with the project's no-sudo ethos). PSU class falls
  out of `hw.model` (Mac Studio vs MacBook Air).
- **Stored fields:** `machine_id`, human label, `fit_type` (`scalar`|`per_rail`), coefficients,
  held-out residual, condition number, meter `{name, tier, accuracy_pct}`, sample/run counts,
  battery version, macOS version, created-at timestamp, `confidence_band`, `tier`.
- **Registry API:** `save(profile)`, `load(machine_id)`, `active_for_current_machine()`, `list()`.
  Graceful miss → `None` (caller falls back to `estimated`).

## 9. Component E — Runtime application

- **Ledger:** add columns `calib_profile_id TEXT`, `calib_confidence TEXT` (additive migration,
  matching the existing `_migrate` pattern). `LedgerRow` and `by_model()` carry them; aggregate
  views stay un-COALESCE'd so an absent rate still surfaces as `None`.
- **Proxy boot:** load `active_for_current_machine()`; hold the profile (or `None`) on app state.
- **`_finalize`:** apply `Cal(e_marginal_rails)` → calibrated marginal joules → `kwh_marginal`;
  stamp `calib_profile_id` + `calib_confidence` (`plug-calibrated (±x%)` or `estimated (±15–30%)`).
  No profile → identity + `estimated`. `Cal` lives in `calibration.py` and is pure.
- **Config:** a `calibration:` block — `auto` (default; load by machine), an explicit `path`, or
  `off`. The existing honesty invariant becomes enforced: **no route may render `calibrated`
  without a matching machine profile** — a violation halts startup with a pointable error.

## 10. Component F — The interactive wizard (`tokenwatt calibrate`)

A guided `typer` flow, re-runnable, with a non-interactive escape hatch later:

1. **Detect machine** — print `machine_id` + human label; note if a profile already exists.
2. **Connect the plug** — prompt for Shelly host/IP (or accept `--meter-host`); verify reachable,
   read the accumulator, and confirm the Mac is the (only) load — a `doctor`-style check that
   fails loud with a fix hint.
3. **Pre-flight** — warn to quiesce the machine (close heavy apps); verify each selected route's
   upstream is reachable; let the user pick which configured routes to calibrate.
4. **Idle baseline** — measure idle rails + idle wall over a fixed window; sanity-check the idle
   wall power is plausible (implausible → "is something else on this plug?").
5. **Run the battery** — for each (route × applicable cell): sustained load for `Δt` with live
   progress (cell, elapsed, tok/s, running wall W). Repeat the whole battery **≥2 passes**.
6. **Fit → gate → band → tier** — show a plain-language result: slope (or per-rail β's), R²,
   held-out residual, condition number, run-to-run agreement, chosen tier, and the achieved band.
7. **Save & activate** — write the profile; offer to set it active.

Plus `tokenwatt calibrate show` (inspect the active profile) and `tokenwatt calibrate --probe`
(C0's raw synchronized-read diagnostic).

## 11. Error handling & failure modes

| Failure | Behavior |
|---|---|
| Shelly unreachable / wrong endpoint | fail loud at connect; offer `--meter manual` |
| No wall delta under load (wrong plug / Mac not on it) | detected via flat accumulator; abort with hint |
| Machine not quiesced / other load on plug | implausible idle → warn, let user retry |
| Inference upstream down | pre-flight abort, point at it (`doctor` style) |
| Ill-conditioned design matrix | stay scalar, log condition number + which rails collapsed |
| Non-repeatable passes | widen band or refuse to certify; never silent |
| Coarse meter cadence vs short window | C0 measures cadence; cell `Δt` chosen above it |

## 12. Testing (intent, not just coverage)

- **`FakeMeterSource`** — a scripted Wh accumulator alongside the existing `FakeMeter`; the whole
  campaign→fit→profile→runtime path runs in CI on no hardware.
- **Fit recovery** — synthetic samples generated from known `a`/`b` (and known `β`) must be
  recovered within tolerance; this fails if the fit math regresses.
- **Honesty gate** — ill-conditioned synthetic data must *refuse* to promote to per-rail and must
  *not* widen-then-claim a tighter band; a Shelly fixture must never produce the lab band.
- **Marginal conservation** — marginal wall ≥ 0 and rail/wall marginals consistent with the
  injected idle baselines.
- **Runtime** — a loaded profile prices a request `plug-calibrated`; an absent profile prices it
  `estimated`; the ledger migration is idempotent.

## 13. Calibration math (reference)

```
# per cell run i, both sides marginal, joules:
E_rail_total,i = Σ_rail E_rail,i                         # E_rail,i already idle-subtracted
E_wall_marg,i  = (Δwh_i · 3600) − P_idle_wall · Δt_i     # Wh→J, minus idle wall

# scalar floor
[a, b]   = NNLS( [E_rail_total | Δt], E_wall_marg )
# per-rail ceiling (gated)
[β]      = NNLS( [E_rail_cpu | … | E_rail_ane | Δt], E_wall_marg )

# runtime, per request:
kWh_marginal = Cal(e_marginal_rails) / 3.6e6
Cal(scalar)  = a · Σ_rail e_rail + b · Δt
Cal(perrail) = Σ_rail β_rail · e_rail + β₀ · Δt
```

## 14. Staged plan — intermediate states to the final state

The **final state is C5**; C0–C4 are independently verifiable intermediate states, and **C2/C4
already deliver a usable scalar calibration** before per-rail lands.

- **C0 — Shelly link + raw probe.** `MeterSource` protocol + `ShellyMeterSource` + plug host in
  config + `doctor`-style reachability + `calibrate --probe` (print synchronized rail Δ / wall Δ
  over a manual window; **measure the meter's effective resolution/cadence**).
  *Verify:* matches the Shelly app under a known load; prove the data is trustworthy first.
- **C1 — Campaign runner + synchronized capture.** Load-cell battery, sustained-load driver to the
  upstream, idle at rails + wall, the `(E_rail[], E_wall, Δt)` sample table.
  *Verify:* energy conservation; deterministic under `FakeMeter` + `FakeMeterSource`.
- **C2 — Scalar fit + profile registry + band/tier.** NNLS scalar fit, `machine_id` detection,
  the JSON profile store, band = residual ⊕ run-variance ⊕ meter accuracy → `plug-calibrated`.
  *Verify:* known `a`/`b` recovered; noisy passes widen the band.
- **C3 — Runtime application.** `calib_*` ledger columns, profile load at boot, `_finalize`
  applies `Cal`, no-match fallback, `calibration:` config block + enforced honesty invariant.
  *Verify:* calibrated rows price with the band; absent profile → `estimated`; migration idempotent.
- **C4 — Interactive wizard.** End-to-end guided `tokenwatt calibrate` + `calibrate show`.
  *Verify:* a full run on the M3 Ultra produces an active profile and the first real
  TokenWatt↔Shelly correlation number.
- **C5 — Per-rail upgrade (ceiling).** Per-rail NNLS + condition/VIF gating + promotion rule.
  *Verify:* the gate refuses an ill-conditioned fit; when it promotes, per-rail beats scalar on
  held-out cells.

## 15. Open questions (deferred — do not block the plan)

- Exact Shelly energy-counter cadence/resolution → answered empirically by C0.
- Minimum sample count + condition-number threshold for per-rail promotion → tuned in C5 against
  real M3 Ultra data.
- Whether bounded concurrency in the load driver improves SNR or just adds attribution noise →
  decided in C1 from observed wall stability.
- Cross-machine profile sharing format/versioning (a shared registry of community profiles) →
  post-loop; the JSON artifact is forward-compatible.
