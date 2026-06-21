# TokenWatt Structured Logging — Design Spec

**Date:** 2026-06-20
**Status:** approved (brainstormed + ratified)

## Purpose

TokenWatt's only durable record today is the **ledger** — a per-request *cost* row written in `_finalize`. But a request that errors or terminates mid-stream may never reach a clean `_finalize`, so it leaves no trace of *why*. That blind spot hid a real failure: a qwen agentic build died mid-stream on its largest tool-call turn, retried 4×, and exited 0/18 through the proxy — with nothing in the ledger explaining the drop.

The **log** is the operational/diagnostic spine that complements the ledger: it records *the path that got there*, including the errored/terminated requests the ledger cannot. Production-grade, not a one-off.

## Principles

- **Module-scoped named loggers** (`tokenwatt.proxy`, `tokenwatt.meter`, `tokenwatt.cli`) — levels controllable per-module; never `print`/bare `logging.*`.
- **Structured JSONL** — one JSON object per line, parseable, diffable across runs, ingestable by any log tool.
- **Correlation** — every request carries a `request_id` (the per-request `label` UUID the proxy already mints) shared across its log lines AND written to its ledger row, so cost ↔ trace join.
- **Privacy by default** — log sizes and token counts, NEVER prompt/response content. (One residual edge: `req.stream_error`/`req.upstream_error` carry `error=str(e)[:300]` — bounded, operator-trusted *transport/network* exception text, not guaranteed content-free; capped at 300 chars. Hardening option in the eventually-bucket.)
- **No new dependency** — stdlib `logging` + a small JSON formatter + `RotatingFileHandler`.
- **Off the hot path** — log at request lifecycle boundaries (start / finish / error), NEVER per streamed chunk at INFO (per-chunk only under DEBUG).

## Event schema

Each event is `{ts, level, logger, event, request_id, ...fields}`. Events:

- **`req.start`** (INFO) — `model`, `req_type`, `stream`, `upstream`, `in_flight`, `body_bytes`
- **`req.finish`** (INFO) — `duration_s`, `ttft_s`, `status`, `tok_in`, `tok_out`, `tok_source`, `marginal_j`, `kwh`, `cost`, `cold`, `aborted`. **`req.finish` means "row booked / energy window closed", NOT "succeeded"** — an aborted stream emits BOTH its abort event AND a `req.finish` with the SAME `request_id`; the `aborted` field (`None` on success, else `"client_disconnect"`/`"stream_error"`) disambiguates, so log analysis counts successes as `aborted == null` without joining events.
- **`req.stream_error`** (WARNING) — **the diagnostic priority**: `error_type`, `error`, `chunks`, `bytes`, `elapsed_s`. Distinguishes **upstream drop** (httpx error — the qwen/LM Studio server cut the stream) from **client disconnect** (`GeneratorExit`/`CancelledError` — pi hung up; logged as `req.client_disconnect` at INFO, NOT an error).
- **`req.upstream_error`** (ERROR) — the 502 path (`client.send`/build failed): `error_type`, `error`.
- **`req.no_route`** (WARNING) — the 404 path: `model`, `routes`.

## Output

- **JSONL → rotating file**, default `~/.tokenwatt/logs/proxy.jsonl` (10 MB × 5 backups).
- **Concise human line → stderr** (live tailing), default on.

## Config surface

`tokenwatt.yaml`:
```yaml
logging:
  level: INFO          # DEBUG|INFO|WARNING|ERROR
  file: ~/.tokenwatt/logs/proxy.jsonl
  console: true        # concise human line to stderr
  max_bytes: 10485760  # 10 MB rotation
  backup_count: 5
```
Plus CLI `--log-level` / `--log-file` (override config) and env `TOKENWATT_LOG_LEVEL`. Precedence: CLI > env > config > default.

## Ledger correlation

`requests` table + `LedgerRow` gain a `request_id TEXT DEFAULT ''` column (forward-only migration), written from the proxy's per-request `label`. Lets a cost row join to its log lines.

## Eventually (out of scope for v1)

- `/metrics` endpoint (Prometheus-style)
- OpenTelemetry / OTLP export
- Log-based alerting
- Async logging (`QueueHandler`/`QueueListener`) if QPS ever outgrows sync file writes
- Per-chunk timing histograms beyond the DEBUG breadcrumb — and when that DEBUG breadcrumb is added, gate it behind `if logger.isEnabledFor(logging.DEBUG):` so the disabled path costs one int comparison per chunk and never formats/writes (keeps the proven-clean streaming hot path clean)
- Harden the `error=` field: log only `error_type` by default, move `str(e)[:300]` behind a DEBUG-gated field

## Acceptance

- A streamed request that the upstream cuts mid-generation produces a `req.stream_error` line with `error_type` + `chunks`/`bytes`/`elapsed_s` — and is attributable to upstream vs client.
- Normal requests produce `req.start` + `req.finish` sharing a `request_id` that also appears on the ledger row.
- Setting `serialize_inference` off (current live mode) is unaffected; logging adds no per-chunk overhead.
- No prompt/response content ever appears in the logs.
