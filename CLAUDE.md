# CLAUDE.md

Guidance for Claude Code (and any AI assistant or contributor) working in this repo.

## What this is

TokenWatt is a transparent, OpenAI-compatible **proxy** that meters the electricity cost of local
LLM inference on Apple Silicon. It forwards each request byte-for-byte to a local inference server,
brackets it with real per-rail SoC energy (Apple IOReport, sudoless), subtracts an idle baseline,
prices it at the user's utility rate, and logs a per-request, per-model ledger.

## Setup & test

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest          # full suite; uses a FakeMeter, so no special hardware needed
```

CI (`.github/workflows/ci.yml`) runs the suite on the supported Python range plus a build /
`twine check` job. Keep it green before merging.

## Architecture (src/tokenwatt/)

- `proxy.py` — the ASGI app + `forward()`: routing, byte-exact streaming passthrough, per-request
  energy bracketing, structured logging, the `_finalize` ledger write. The most load-bearing file.
- `meter.py` — `EnergyByRail`, `ZeusMeter` (real IOReport), `FakeMeter` (deterministic test double).
- `idle.py` — `IdleBaseline` (rolling-median idle power, gated while requests are in flight).
- `ledger.py` — sqlite ledger, `LedgerRow`, `by_model()` rollups (incl. `$/Mtok`).
- `cloud.py` — dated, editable cloud price table + total-cost comparison.
- `cli.py` — `typer` CLI: `serve` / `report` / `compare` / `wrap` / `init`, and display helpers.
- `config.py` / `router.py` — pydantic config + YAML loader; model→upstream routing (exact > glob > `*`).
- `usage.py` / `coldstart.py` / `reqtype.py` / `log.py` / `rate.py` — token counting, model-load
  (cold-start) booking, request-type classification, JSONL logging, and rate pricing.

## Conventions that matter

- **Honesty contract** — never a fabricated number. Costs are labeled `estimated (±15–30%)` until
  wall-calibrated. A model with no rate renders `—`, a real sub-cent cost renders `<$0.0001`, and a
  missing per-token figure renders `-` — never a fake `$0`/`0`. Cost SUMs stay un-COALESCE'd so an
  absent rate surfaces as `None`, not `0`.
- **The proxy is a meter, not a gateway** — byte-exact passthrough, no buffering, no rewriting.
  Streaming, tool calls, sampling params, chat templates, and `response_format` pass straight through.
- **Apple Silicon only** for the real meter (`zeus-apple-silicon` / IOReport), gated by a platform
  marker so `pip install` still works elsewhere. Tests use `FakeMeter`, so they run anywhere.
- **Tests encode intent** — a test should fail when the behavior it covers breaks, not merely when
  code is absent. Energy-conservation and the honesty rules are explicitly asserted.
- `VERSION` auto-increments its patch on every commit via `.githooks/pre-commit` — expected; let it run.

## Docs

Design specs and milestone implementation plans live under `docs/design/`.
