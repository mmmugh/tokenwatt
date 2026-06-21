# TokenWatt — Design Spec

- **Status:** Draft for review
- **Date:** 2026-06-19
- **Author:** Justin Stewart (with Claude)
- **Working name:** TokenWatt _(binds tokens ↔ watts — the core mechanic; alternatives: WattForward, SipWatt, kWhisper, Coulomb)_

A local HTTP proxy for Apple Silicon Macs that sits in front of OpenAI-compatible local
inference servers, brackets each request with real per-rail SoC energy measurement, and
converts it to a **wall-calibrated, time-of-use-aware electricity cost** — logged as a
per-request, per-model ledger. The point is to **know your real number**, not to prove
local inference is expensive (on Apple Silicon it is often "effectively free", and the tool
is valuable whichever way the number falls).

---

## Defaults taken (flag any to change at review)

| Decision | Default | Change to… |
|---|---|---|
| Name | **TokenWatt** | any shortlist name |
| Headline `$` | **marginal** (idle-subtracted) only in v1 | add total / amortized |
| Cost surfacing | **log-only** (no in-response headers) | opt-in `x-energy-*` headers |
| Carbon (gCO2e) | **deferred**, reserved ledger column | pull into v1 |

---

## 1. Problem & goal

Running LLM inference locally has **no per-token price but a real electricity cost**. No
maintained tool today glues *Apple-Silicon energy-per-live-request* to a *real residential /
time-of-use electricity-bill dollar*. TokenWatt does exactly that.

**Primary goal:** a **precise, wall-calibrated per-request electricity ledger** for local
inference on Apple Silicon — accurate to a few percent absolute under serialized single-tenant
conditions, with per-model / per-request-type breakdown and TOU-aware pricing.

**Success criteria**
- Every request through the proxy produces a ledger row with measured energy (per rail),
  joules, kWh, and marginal `$` at the correct TOU rate.
- With a calibration profile loaded, absolute energy is within **±2–5%** of a wall meter;
  without one, numbers are clearly labeled **"estimated, ±15–30%"** — never silently.
- Per-model comparison answers "what does m1 vs v1 cost per token" in measured numbers.
- Forwarding is byte-exact: the proxy is a transparent drop-in; clients see no behavior change.

## 2. Non-goals

- **Dialect translation** (Anthropic→OpenAI etc.) — that is CCR / LiteLLM's job. TokenWatt
  *accepts and passes through* dialects; it never rewrites bodies.
- **Cloud-provider cost / multi-provider routing** — TokenWatt measures *local electricity*.
- **Per-PID hardware watts** — impossible on Apple Silicon; see §14.
- **Full utility bill** — demand charges and stateful tiered blocks are out (see §9). This is
  an **energy-cost** estimator, not a bill reproducer.
- **Non-Apple-Silicon platforms.**

## 3. v1 scope

**Core (always on):** thin ASGI proxy · multi-backend routing (`model → upstream`) ·
OpenAI ingress (`/v1/chat/completions`, `/v1/embeddings`, `/v1/responses`) · byte-exact
streaming passthrough (SSE) · zeus per-rail energy window + idle-baseline subtraction ·
per-request-type ledger (generative / vision / embedding) in sqlite · per-model rollups ·
**marginal** cost · backend-`usage` extraction with self-count fallback · CLI report ·
honest confidence labeling · dialect-agnostic **energy-only fallback** for unrecognized paths.

**Promoted into v1 (this scope decision):**
- **#2 TOU rate model** — URDB-structured engine; flat `$/kWh` is the degenerate case.
- **#3 Wall-calibration profiles + confidence tiers** — per-machine regression; labels every number.
- **#4 Self-calibration wizard** — guided campaign that *produces* a profile, with manual /
  smart-plug / **lab-meter** input modes (a lab-grade power analyzer drives this).

**Deferred (eventually, not v1):** in-response `x-energy-*` headers · precision-serialize
mode · Ollama-native + Anthropic ingress adapters · total & amortized cost numbers ·
break-even-vs-cloud reporting · menu-bar app · overlap apportionment math · carbon/gCO2e ·
dialect translation · cross-machine profile prediction · image/audio-endpoint accounting ·
tiered/block pricing.

> The architecture (§5) keeps seams for every deferred item so adding them later does not
> touch the measurement core.

## 4. Scope rationale (why this, why dialect-ready)

- **Demand is real but the framing is "know your real number."** "Is local cheaper than the
  API once I count electricity?" is a top recurring r/LocalLLaMA / HN question, always answered
  with napkin spreadsheets that *guess* wattage and tok/s. A measured proxy removes the guess.
  (Ollama issue #16339 requested this and was closed "not planned" — the niche is open.)
- **Dialect landscape (why v1 is OpenAI but the design is dialect-ready):** server-side
  OpenAI-compat is effectively universal (**0–5%** of popular servers lack it; 0/16 in survey).
  Client/traffic-side non-OpenAI is **25–60%** (scope-dependent, low confidence — no vendor
  publishes per-endpoint telemetry), dominated by **Ollama-native** with **Anthropic
  `/v1/messages`** small-but-fastest-growing (40%+ inside the coding-agent niche). v1 ships
  OpenAI ingress (covers the user's three backends and the majority of traffic) plus an
  energy-only fallback that already meters everything; Ollama-native + Anthropic adapters are a
  cheap post-v1 add because **energy measurement is dialect-agnostic; only token extraction is
  per-dialect.**

## 5. Architecture

```
 clients (OpenAI dialect in v1; Ollama/Anthropic ready post-v1)
        │  /v1/chat/completions · /v1/embeddings · /v1/responses
        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ TokenWatt — thin ASGI proxy, one local port                           │
 │                                                                        │
 │ ① INGRESS ADAPTERS  (dialect-aware, thin)                             │
 │     • classify dialect + request-type (generative│vision│embedding)   │
 │     • read `model` → route lookup → upstream {host:port, dialect}     │
 │ ──────────────────────────────────────────────────────────────────── │
 │ ② MEASURE t0   zeus per-rail energy counters   ◄── dialect-AGNOSTIC   │
 │ ③ FORWARD      httpx stream, bytes UNCHANGED (SSE preserved)          │
 │ ④ TEE          copy stream → usage extractor (dialect+type) │ or none │
 │ ⑤ MEASURE t1   ΔE/rail − idle baseline → calibration profile → kWh    │
 │ ⑥ PRICE        kWh × TOU rate(t)  →  marginal $                       │
 │ ⑦ EMIT         ledger row (sqlite)                                    │
 └───┬───────────────────────┬───────────────────────┬──────────────────┘
     ▼                       ▼                        ▼
 m1 mlx-openai-server    v1 mlx-vlm              embeddings
   (OpenAI; text)        (OpenAI/Anthropic; vision)   (mlx-openai-server)

 in-process services:
   • zeus sampler          continuous idle baseline + per-request windows
   • usage-extractor reg.  {OpenAI[+Ollama,Anthropic]} × {gen│vision│embed}
   • calibration registry  per-machine profile + confidence band   (#3)
   • self-calibration      campaign runner + meter sources          (#4)
   • rate model            TOU/flat, local URDB cache, ts→$/kWh     (#2)
   • ledger store          sqlite: per-request / per-model / per-type / per-rail
   • reporting             CLI: today/month $, per-model J/token, $/Mtok, tok/s
```

**Internal contract:** every request reduces to a normalized pair
`(TokenUsage, EnergyByRail)` that all downstream services consume — so dialect, request type,
and backend identity never leak past the ingress/measurement boundary. This is the seam that
makes deferred features additive.

## 6. Component A — Proxy & ingress

- **Single local port**, thin ASGI app (FastAPI/Starlette + `httpx.AsyncClient`).
- **Routing:** `model → upstream {host, port, dialect, role}` config table. The `model` field
  is present in all dialects, so one routing key works everywhere. Default is **same-dialect
  passthrough** (no translation).
- **Byte-exact streaming:** forward upstream bytes **unchanged** (do NOT parse-then-reserialize
  — it corrupts SSE framing, keepalives, tool-call deltas). A side tee copies chunks to the
  usage extractor. Handle the `[DONE]` sentinel, client disconnect (cancel upstream), and
  missing usage chunks.
- **Request-type classifier:** vision = image content parts present; embedding = embeddings
  route; else generative. Determines the ledger denominator and which extractor runs.
- **Usage-extractor registry** — one parser per `(dialect × type)`, each returning
  `TokenUsage{input, output, cached, source, confidence}` where `source ∈
  {backend, self-count, count_tokens-endpoint}`:
  - OpenAI chat → `usage.{prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens}`
  - OpenAI embeddings → `usage.prompt_tokens` (input only; no completion)
  - Vision → trust backend `usage`; if absent, query mlx-vlm `/v1/messages/count_tokens`;
    else report energy-per-image + per output token, `confidence=low`
  - (post-v1) Anthropic → `usage.{input_tokens, output_tokens, cache_read_input_tokens}`;
    Ollama → `prompt_eval_count`/`eval_count` (+ durations → tok/s)
- **Self-count fallback (mandatory for streaming):** mlx-openai-server exposes **no**
  `stream_options.include_usage`, so streaming chat may omit a final usage chunk → count
  completion tokens from the teed deltas (tiktoken / model tokenizer), `source=self-count`.
- **Energy-only fallback:** unrecognized path/dialect → still bracket energy, emit row with
  `TokenUsage=null`, `confidence=energy-only`. Reach is never gated on having a parser.

## 7. Component B — Measurement

Built on **`zeus-apple-silicon`** (ml-energy) reading Apple's IOReport "Energy Model" channel —
**sudoless**, ~1 mJ resolution, windowed counter-deltas. `powermetrics` is a **bench
cross-check only** (root-only, 5 s grid; not in the runtime path).

- **Window:** read cumulative per-rail energy counters at request ingress (`t0`) and at stream
  completion / `[DONE]` (`t1`). `ΔE_rail = counter_rail(t1) − counter_rail(t0)` **is** the joules
  for exactly that window — no fixed-grid integration.
- **Rails (M2+, incl. M3 Ultra):** ECPU, PCPU, CPU-total, GPU, GPU-SRAM, **DRAM**, ANE.
  Include **DRAM** explicitly — MLX decode is memory-bandwidth-bound and CPU+GPU+ANE alone
  undercounts generation by ~10–30%. (M1 omits DRAM/ANE/GPU-SRAM → scalar calibration; not the
  primary target.)
- **Idle baseline:** a background sampler records per-rail idle power `P_idle,rail` whenever no
  request is in flight (rolling median). The idle energy over the window is
  `E_idle,rail = P_idle,rail · Δt`.
- **Two-tier accuracy** (falls out of whole-SoC measurement; no apportionment math in v1):
  - **Aggregate / per-model / per-window energy is always exact** — the SoC is fully metered.
  - **Per-request attribution** is exact when the request had the SoC to itself
    (`attribution=solo`); when windows overlap (concurrent requests, or mlx-openai-server's
    internal batch scheduler), the row is still emitted with whole-window energy but flagged
    `attribution=overlap, confidence=low`. (Splitting overlap energy = deferred #10.)
- **Cold-start booking:** the first request to a just-loaded/swapped model pays weight-load
  energy (seconds of DRAM/IO) unrelated to the prompt. Detect via anomalous pre-first-token
  energy / TTFT and book it to a separate `model_load` event row, **never** the triggering
  request. With three resident models on the Ultra's large unified memory this is mostly a
  one-time startup cost.

## 8. Component C — Calibration (#3 + #4)

Turns zeus's **modeled** on-die estimate (Apple's CLPC Digital Power Estimator) into a
**wall-calibrated measurement** with a stated error band.

- **Model (M2+ full rails):** multivariable linear regression of measured AC wall energy on
  per-rail zeus energy over a battery of workloads:
  `E_wall ≈ β₀·Δt + Σ_rail β_rail · E_rail`. The β coefficients absorb both the DPE offset and
  PSU AC/DC loss, **per rail**, so the fit stays valid as workloads shift between compute-bound
  prefill (GPU-heavy) and memory-bound decode (DRAM-heavy). (M1: scalar `E_wall ≈ a·E_combined + b`.)
- **Profile registry:** profiles keyed by **machine id (SoC + model + PSU class)**, each storing
  β, fit residual, meter accuracy, sample count, macOS version, and a derived **confidence band**.
  Lookup at runtime; **graceful fallback** to raw-uncalibrated zeus labeled
  `confidence=estimated (±15–30%)` for unknown machines.
- **Confidence tiers** stamped on every ledger row: `calibrated (±2–5%)` when a matching profile
  is loaded, else `estimated (±15–30%)`. Never present uncalibrated as calibrated.
- **Self-calibration wizard (#4):** a guided campaign that runs a fixed workload battery at
  varied load (idle, prefill-heavy, decode-heavy, vision, embeddings), captures
  `(E_rail, E_wall)` pairs, fits β, computes residual → band, and saves the profile.
  - **Meter sources** via a `MeterSource` interface: **manual** entry (type wall-watt readings
    at prompts — works with any Kill-A-Watt), **smart-plug** (TP-Link Kasa/P110 local API), and
    **lab-meter** mode (serial/CSV ingest from a lab-grade power analyzer).
    v1 ships manual + a documented extension point; smart-plug/lab adapters are thin additions.
  - Calibration uses the **marginal** convention end-to-end: subtract a measured idle baseline at
    the wall too (`E_wall_marginal = E_wall_active − E_wall_idle`) so the fit maps marginal→marginal.

## 9. Component D — Rate model (#2)

One internal structure where **flat is the degenerate case of TOU** — adopted from the NREL
OpenEI URDB energy-charge subset:

```
RateModel {
  periods: [ [ {rate, adj, max?} ...tiers ] ...periods ],   // $/kWh + rider adj per tier
  weekday_schedule: int[12][24],   // month × hour → period index
  weekend_schedule: int[12][24],
  fixed_monthly?: float            // informational; not attributed per request
}
price(kWh, t):
  p = (is_weekend(t) ? weekend_schedule : weekday_schedule)[t.month-1][t.hour]
  tier = periods[p][0]                       // tier 0 (see omissions)
  return kWh * (tier.rate + tier.adj)
```

- **Flat `$0.31`** → one period, one tier, all-zero schedules. **Custom TOU** → peak/off-peak
  rates + weekday hour windows compiled to the same `periods` + 12×24 matrices. **URDB lookup**
  by zip/utility populates it automatically.
- **Local cache** of the URDB energy-charge subset (`usurdb` gz) → no network call at inference
  time; attribute NREL/OpenEI.
- **Omissions (stated honestly):** demand charges (`$/kW` on a 15-min peak — unattributable to a
  sub-kWh request) are dropped; tiered/block pricing is stateful, so v1 defaults to **tier 0**
  (exact for most single-tier residential TOU defaults); no holiday calendar / real-time pricing;
  US-only auto-lookup (non-US users use the custom-TOU form). Requests spanning a TOU boundary use
  the **start-time** period in v1.

## 10. Component E — Ledger schema (sqlite)

`requests` (one row per forwarded request):

| column | meaning |
|---|---|
| `id, ts_start, ts_end, duration_ms, ttft_ms` | identity + timing |
| `dialect, model, upstream, req_type` | routing + `gen│vision│embed` |
| `tok_in, tok_out, tok_cached, tok_source, tok_confidence` | normalized `TokenUsage` |
| `e_ecpu_j … e_dram_j … e_ane_j` | per-rail joules (window Δ) |
| `e_window_j, e_idle_j, e_marginal_j` | summed window / idle / marginal joules |
| `energy_source` | `zeus` (always in v1) |
| `kwh_marginal` | calibrated marginal energy → kWh |
| `calib_profile_id, calib_confidence` | `calibrated(±x)` / `estimated(±20%)` |
| `rate_period, rate_usd_kwh` | TOU period + effective rate at `ts_start` |
| `cost_marginal_usd` | the hero number |
| `attribution, overlap_flag` | `solo` / `overlap` |
| `tok_per_s` | throughput |
| `gco2e` | reserved, null in v1 |

`model_load` (separate): `id, ts, model, upstream, e_j, duration_ms, trigger`.

**Rollup views:** per-model, per-type, per-day — exposing J/token, `$/Mtok`, tok/s, and
total kWh / `$`. Aggregate views never depend on per-request `attribution` confidence.

## 11. Component F — Reporting (CLI, v1)

- `tokenwatt serve` — run the proxy.
- `tokenwatt report` — today/this-month total kWh + `$`; per-model table (J/token, `$/Mtok`,
  tok/s, request count); confidence banner if any rows are `estimated`.
- `tokenwatt calibrate` — launch the self-calibration wizard (#4).
- `tokenwatt rate` — set flat `$/kWh`, import a custom TOU schedule, or URDB lookup.

## 12. Component G — Configuration

Single file, **YAML** (not TOML — the config is dominated by the `routes` list-of-records, where
YAML's `- name:` list stays scannable at 3 or 30 backends; the ecosystem's reference configs —
mlx-openai-server, LiteLLM — are YAML, lowering surprise).

- **Zero-config boot (Caddy philosophy):** `tokenwatt serve` with no file boots and measures —
  port **7000**, host `127.0.0.1`, ledger `~/.tokenwatt/ledger.sqlite`, rate falling back to a
  labeled **`estimated`** number, calibration `auto`, default localhost passthrough route. Config
  exists only to add routes and a real `$/kWh` — never to make the tool start. Smallest useful
  config is ~15–20 lines.
- **Two sections:** a flat `server` block (port, host, ledger, rate, calibration) set once, then a
  `routes:` list you iterate on. No semantic task-class router — TokenWatt routes by the literal
  `model` field.
- **Validation (fail-loud, Pydantic):** a bad config dies *before the proxy binds*, naming the
  field path + bad value + fix hint, e.g.
  `routes[2].type: 'embedding' invalid (expected text|vision|embeddings) — did you mean 'embeddings'?`.
  Cross-field invariants halt startup: duplicate route names, non-URL upstreams, and — the honesty
  contract — a route may not be stamped `calibrated` without a matching machine profile. A
  misrouted backend poisons the ledger, so silent validation is forbidden.
- **Scaffolding:** `tokenwatt init -c tw.yaml --example m1v1emb` writes a fully-commented config
  (documentation that can't drift — it *is* the schema instantiated); non-destructive (no overwrite
  without `--force`); shipped as `examples/m1-v1-embeddings.yaml`.
- **Routing (load-bearing):** each route's `match:` list holds **both** a short alias (`m1`) **and**
  the full HF id / glob (`mlx-community/Qwen3-*`) — clients send either. Precedence is explicit:
  **exact > longest-prefix/glob > `*` catch-all**, first-in-list breaking ties. The route `name` is
  the **canonical key** ledger and reports group by, so "m1 vs v1" stays clean regardless of the
  40-char id sent. When a model matches **no** route and no `*` catch-all is configured, the proxy
  returns a **fail-loud 404** listing the configured routes — never a silent forward to a wrong
  backend (which would poison the ledger). Zero-config ships a `*` catch-all, so the default
  experience never 404s; add a `match: ["*"]` route for transparent catch-all passthrough. (The
  *energy-only fallback* in §6 is the separate, dialect-level fallback that applies on a **matched**
  route whose response usage can't be parsed — not a routing fallback.) No regex DSL; no env-var
  interpolation in v1 (localhost upstreams, no secrets).

```yaml
# examples/m1-v1-embeddings.yaml — run:  tokenwatt serve -c tw.yaml
# Every key is optional. With NO config, `tokenwatt serve` still boots and
# measures (port 7000, ~/.tokenwatt/ledger.sqlite, estimated rate, auto
# calibration, localhost passthrough). Config only adds routes + a real rate.

port: 7000                          # local proxy listen port (default)
host: 127.0.0.1                     # loopback only by default
ledger: ~/.tokenwatt/ledger.sqlite

rate:
  flat_usd_per_kwh: 0.31            # omit entirely -> numbers labeled 'estimated'
  # tou:                           # future: time-of-use periods drive $/kWh by hour
  #   urdb_zip: 94110

calibration:
  profile: auto                    # auto | <profile-id> | none
  # auto = match this machine; unknown -> rows stamped 'estimated (±15-30%)',
  #        NEVER silently promoted to 'calibrated'.

# model -> upstream routes (the spine).
# Match precedence: exact > longest-prefix/glob > '*'; first-in-list breaks ties.
routes:
  - name: m1                       # canonical id used in ledger + report
    type: text                     # text | vision | embeddings -> ledger denominator
    upstream: http://127.0.0.1:8081      # mlx-openai-server
    match:
      - m1                                      # short alias
      - mlx-community/Qwen3-Coder-Next-4bit      # full HF id the OpenAI SDK sends
      - mlx-community/Qwen3-*                     # glob: survive quant/version bumps

  - name: v1
    type: vision
    upstream: http://127.0.0.1:8082      # mlx-vlm
    dialect: openai                # openai (default) | anthropic — affects token
                                   # extraction only, not energy. (mlx-vlm speaks both.)
    match:
      - v1
      - mlx-community/Qwen2-VL-*

  - name: embeddings
    type: embeddings
    upstream: http://127.0.0.1:8081      # same mlx-openai-server process as m1
    match:
      - embeddings
      - text-embedding-*
      - mlx-community/*-embed-*
```

## 13. Cost math

For window `[t0, t1]`, `Δt = t1 − t0`, rails `r`:

```
E_window  = Σ_r ΔE_r                                  # measured joules in window
E_idle    = Σ_r P_idle,r · Δt                         # idle energy over same window
E_wall_window = Cal(ΔE_r, Δt)                          # apply β profile to window rails
E_wall_idle   = Cal(P_idle,r·Δt, Δt)                   # apply β profile to idle-equiv rails
E_marginal_wall = E_wall_window − E_wall_idle          # calibrated marginal AC energy
kWh_marginal    = E_marginal_wall / 3.6e6
cost_marginal   = kWh_marginal × price(kWh=1, t0)      # TOU rate at request start
```

`Cal` is the regression of §8 (scalar on M1). Uncalibrated runs use `Cal = identity` and stamp
`estimated`. (Deferred: `cost_total = E_wall_window @ rate`; `cost_amortized = cost_marginal +
purchase_price/lifetime_tokens × tok_out`.)

## 14. Accuracy & honesty model

- **No per-PID watts exist on Apple Silicon.** The one per-PID energy API
  (`proc_pid_rusage.ri_energy_nj`) is CPU-cores only — blind to GPU/ANE/DRAM where MLX lives;
  `powermetrics --show-process-energy` is a unitless score, **not** billable. Honest per-request
  attribution = **whole-SoC window − idle baseline**, valid when inference is the dominant load.
- **Two independent error axes:** (1) *magnitude* — zeus is a modeled DPE estimate ~10–15% off a
  wall meter, +~10–15% PSU loss; **calibration (§8) removes this** → ±2–5%. (2) *attribution* —
  overlap/contention; handled by the `solo`/`overlap` flag and the always-exact aggregate.
- **Always emit the confidence band**; present ranges, not false precision.
- **Reframe:** on Apple Silicon the answer is often "effectively free" (~$4.50/mo on an M4 Max).
  Lead reporting with **per-model comparison** and the **measured number + band**, not a scary
  absolute dollar. The tool wins whichever way the number falls.

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| IOReport is a private framework; could break on a macOS update | Pin to zeus; `powermetrics` documented as fallback; record macOS version in profiles |
| Whole-SoC attribution smears under contention | `solo`/`overlap` flag; aggregate stays exact; precision-serialize mode available later |
| Streaming `usage` absent (mlx-openai-server) | Mandatory self-count from teed deltas |
| Vision token counts hard to self-derive | Trust backend `usage` / `count_tokens` endpoint; else energy-per-image + `confidence=low` |
| Absolute error band undercuts "precise" goal | Wall-calibration (lab-grade analyzer) → ±2–5%; honest tiers otherwise |
| TOU/URDB scope creep | Energy-charge subset only; tier 0; demand charges dropped |
| Closest tool (llmtop) is adding Apple Silicon | Moat = real utility-rate / TOU electricity-bill `$` on local, wall-calibrated — none of the prior art does this combination |

## 16. Build order (milestones)

Each milestone is independently verifiable, and each carries the adoption work that makes that
slice *usable*, not merely functional (see §17).

- **M0 — End-to-end core + frictionless first run (M3 Ultra).** Thin OpenAI passthrough (one
  backend) + zeus window + idle baseline + sqlite ledger + flat-rate marginal `$` + `report`.
  *Adoption:* `uv tool install` / `uvx` packaging with a console entry point; **zero-config
  `tokenwatt serve`** that boots and measures; **no sudo**; `uvx --from git+…@TAG` try-now line.
  *Verify:* energy vs a manual wall-meter reading; `tokenwatt serve` measures with no flags.
- **M1 — Multi-backend + types + real config.** `model→upstream` routing; embeddings + vision
  request-types; self-count; per-model rollups; energy-only fallback; cold-start booking.
  *Adoption:* **YAML config + Pydantic fail-loud validation + `tokenwatt init`** scaffold +
  dual-match routing; `examples/m1-v1-embeddings.yaml`; the README **hero GIF**.
  *Verify:* ledger vs known token counts; a bad config halts with a pointable error.
- **M2 — Calibration loop (#3 + #4) + machine-readable + releases.** Per-rail regression; profile
  registry + confidence tiers; self-calibration wizard (manual + lab-meter modes); run the
  lab-grade calibration campaign. *Adoption:* `--json` / `--once` output; tag-driven release pipeline (OIDC
  trusted publishing). *Verify:* calibrated band vs lab ground truth; `--once` emits valid JSON.
- **M3 — TOU rate model (#2) + the shareable card + menu-bar.** URDB cache + custom-TOU form +
  flat; `ts→$/kWh`; `rate` CLI. *Adoption:* `tokenwatt wrap` **verdict card** (credible now that
  numbers are calibrated *and* rate-accurate); **SwiftBar** menu-bar plugin.
  *Verify:* a TOU request prices at the correct period; `wrap` produces a shareable PNG + markdown.

**Repo conventions:** `VERSION` starts `0.1.0`, auto-increments patch per commit via
`.githooks/pre-commit`; feature work happens on branches, not `main`.

**post-v1:** notarized standalone `.app`; Homebrew tap; Prometheus exporter; Ollama-native +
Anthropic ingress adapters; total/amortized cost numbers; carbon (gCO2e).

## 17. Adoption, packaging & shareability

"Will people download and use this" is a v1 design constraint. The bar: **one command to install,
useful on first run, one command to share the result.**

- **Install (the funnel) — match the user's existing ritual.** Publish to PyPI with a
  `[project.scripts]` entry point so `uv tool install tokenwatt` and `uvx tokenwatt` work — the same
  verb the user already runs for mlx-openai-server, zero new tooling. Document
  `uvx --from git+https://github.com/<you>/tokenwatt@<TAG> tokenwatt` as the install-free "try it
  now" / pre-PyPI channel.
- **Time-to-first-value.** Bare `tokenwatt serve` measures with no config, no API key, **no sudo**
  (the runtime reads IOReport-style native APIs like macmon/mactop; only the *optional*
  `powermetrics` bench cross-check ever touches root). "No sudo" goes in the README's first line —
  per-launch password prompts (`sudo asitop`) are the category's #1 first-run killer and our wedge.
- **The shareable hook.** `tokenwatt wrap` emits a 30-day **"my inference bill vs. what cloud would
  have cost" verdict card** — a self-contained PNG + copy-pasteable markdown with pre-filled share
  text, fusing (1) the surprising first-person dollar figure ("I ran inference on my M3 Ultra for 30
  days. It cost $6.10.") and (2) the break-even punchline *including the contrarian inverse* ("Cloud
  Sonnet would've cost $213 — this Mac pays for itself in 9 months" / "At your usage, local never
  beats the API"). The Mac model + token volume is stamped on the card for identity attachment. The
  per-model `$/Mtok` table and the menu-bar live-wattage number are **supporting features, never the
  hook** (the table is commoditized; the live number rides in screenshots but doesn't start threads).
- **README order (image-first):** live-cost GIF → one-sentence value prop with the wedge
  ("OpenAI-compatible proxy that measures what local inference actually costs you in electricity, per
  model, per request — on Apple Silicon, no sudo") → one-command install → bare run → the verdict
  card → `tokenwatt wrap` → add-your-backends (`tw.yaml` + `init`) → keybindings / `--json` →
  calibration & the honesty contract → how-it-works / methodology last.
- **Packaging & release.** Primary channel PyPI + uv. Menu-bar ships first as a **SwiftBar plugin**
  (a Python script shelling out to the installed CLI) — SwiftBar is already notarized, so no $99/yr
  Developer ID, codesign, or notarization; a standalone `rumps`+`py2app` notarized `.app` is
  **post-v1**, only after non-CLI demand (Sequoia/Tahoe 26 removed the Control-click Gatekeeper
  bypass, so unsigned `.app`s now fail). Homebrew tap deferred to proven demand.
- **Versioning.** Per the maintainer's "nothing is ever finished" workflow, `VERSION`
  auto-increments the **patch tier on every commit** (current `.githooks/pre-commit`), starting
  `0.1.0`; releases tag the `VERSION` at that commit and publish via **OIDC trusted publishing** (no
  stored token). _Alternative considered:_ Conventional-Commits-driven SemVer via release-please
  (feat→minor, fix→patch); deferred in favor of the simpler per-commit scheme — revisit if
  minor/major semantics become useful.
- **Do-not-over-build (YAGNI guardrails):** no semantic task-class router; no LiteLLM-style
  load-balancing / fallbacks / retries / redis; no env-var interpolation; no regex match DSL; config
  never mandatory to boot; never auto-promote uncalibrated numbers to `calibrated`; no
  git-clone+build install tier; no per-launch sudo; no API-key / account gate on first run; menu-bar
  stays one ambient number (no day-one analytics app); no bespoke `curl | bash` installer.

## 18. Open questions (deferred — do not block v1)

- Distribution: free OSS / paid menu-bar (TokenBar-style) / research library.
- Whether to ship a cross-machine profile *prediction* (PSU + SoC-gen) once enough lab anchors exist.
- Default-on vs opt-in for precision-serialize once it lands.

## Appendix — verified backend facts (installed source, 2026-06-19)

- **mlx-openai-server v1.8.1** (`app/api/endpoints.py`): full OpenAI surface
  (`/v1/chat/completions`, `/v1/embeddings`, `/v1/responses`, images, audio, `/v1/models`,
  `/v1/queue/stats`); serves text/vision/embeddings/audio itself; reports `usage` incl.
  `prompt_tokens_details.cached_tokens`; **no** `stream_options.include_usage`; internal batch
  scheduler (can overlap requests).
- **mlx-vlm server** (`mlx_vlm/server/app.py`, `openai.py` + `anthropic.py`): OpenAI **and**
  Anthropic dialects; `/v1/chat/completions`, `/v1/messages`, **`/v1/messages/count_tokens`**,
  `/v1/cache/*`, `/unload`.

**Key sources:** `zeus-apple-silicon` (github.com/ml-energy/zeus-apple-silicon) · NREL OpenEI
URDB (openei.org/services) · closest prior art: llmtop, Neuralwatt, Stanford
Intelligence-Per-Watt, LiteLLM (none combine local Apple-Silicon energy + real TOU utility rate).
