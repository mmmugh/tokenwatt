# TokenWatt Structured Logging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a production-grade structured logging subsystem to TokenWatt — module-scoped named loggers emitting JSONL request-lifecycle events (incl. a stream-error capture that distinguishes upstream drop from client disconnect) to a rotating file + a concise console line, with a `request_id` correlating each ledger row to its log lines.

**Architecture:** A new `tokenwatt/log.py` provides a JSON-line formatter, a concise console formatter, an `event()` helper, and `setup_logging()`. `config.py` gains a `LoggingConfig`. `ledger.py` gains a `request_id` column. `proxy.py` emits the lifecycle events keyed by the per-request `label` UUID. `cli.py serve` calls `setup_logging()` with CLI/env/config precedence.

**Tech Stack:** Python 3.12, stdlib `logging` + `logging.handlers.RotatingFileHandler`. No new dependency.

## Global Constraints

- **No new dependency.** stdlib only.
- **Privacy:** NEVER log prompt/response content — only sizes (`body_bytes`, `chunks`, `bytes`) and token counts.
- **Off the hot path:** log only at lifecycle boundaries (start / finish / error). NO per-chunk logging at INFO (DEBUG only, and not in v1).
- **Correlation id:** reuse the proxy's existing per-request `label` (a uuid4 hex) as `request_id`, on both the log events and the ledger row.
- **`GeneratorExit`/`asyncio.CancelledError` are NOT errors** — they are client disconnects; log at INFO as `req.client_disconnect` and re-raise. They are `BaseException`, so they must be caught in an `except (GeneratorExit, asyncio.CancelledError)` clause placed BEFORE `except Exception`.
- **Idempotent setup:** `setup_logging()` clears existing handlers so re-invocation doesn't duplicate output.
- **Defaults:** level INFO, file `~/.tokenwatt/logs/proxy.jsonl`, rotation 10 MB × 5, console on.
- **Precedence:** CLI `--log-level`/`--log-file` > env `TOKENWATT_LOG_LEVEL` > config `logging:` > built-in default.

---

## File Structure

```
src/tokenwatt/log.py        # NEW: formatters, event(), setup_logging()
src/tokenwatt/config.py     # MODIFIED: LoggingConfig + Config.logging
src/tokenwatt/ledger.py     # MODIFIED: request_id column + LedgerRow field + migration
src/tokenwatt/proxy.py      # MODIFIED: lifecycle event logging + request_id to ledger
src/tokenwatt/cli.py        # MODIFIED: serve calls setup_logging w/ precedence + flags
tests/test_log.py           # NEW
tests/test_config_logging.py# NEW
tests/test_ledger.py        # MODIFIED
tests/test_proxy.py         # MODIFIED
tests/test_cli_config.py    # MODIFIED (or new test for precedence)
```

---

### Task 1: `tokenwatt/log.py` — formatters + setup

**Files:** Create `src/tokenwatt/log.py`, `tests/test_log.py`.

**Interfaces — Produces:**
- `event(**fields) -> dict` — returns `{"tw": fields}` for use as `logger.info("req.start", extra=event(...))`.
- `JsonLineFormatter` / `ConsoleFormatter` (logging.Formatter subclasses).
- `setup_logging(*, level="INFO", file="~/.tokenwatt/logs/proxy.jsonl", console=True, max_bytes=10_485_760, backup_count=5) -> None` — configures the `tokenwatt` logger tree (idempotent).

- [ ] **Step 1: Failing tests** (`tests/test_log.py`)

```python
import json
import logging
from tokenwatt.log import event, JsonLineFormatter, setup_logging


def test_event_wraps_fields():
    assert event(model="m1", in_flight=1) == {"tw": {"model": "m1", "in_flight": 1}}


def test_json_formatter_emits_one_parseable_line():
    rec = logging.LogRecord("tokenwatt.proxy", logging.INFO, "f", 1, "req.start", None, None)
    rec.tw = {"model": "m1", "body_bytes": 42}
    line = JsonLineFormatter().format(rec)
    obj = json.loads(line)                       # one valid JSON object
    assert obj["event"] == "req.start" and obj["level"] == "INFO"
    assert obj["logger"] == "tokenwatt.proxy" and obj["model"] == "m1" and obj["body_bytes"] == 42
    assert "ts" in obj


def test_setup_logging_writes_jsonl_and_is_idempotent(tmp_path):
    f = str(tmp_path / "logs" / "proxy.jsonl")
    setup_logging(level="INFO", file=f, console=False)
    setup_logging(level="INFO", file=f, console=False)   # second call must not duplicate handlers
    root = logging.getLogger("tokenwatt")
    assert len(root.handlers) == 1                        # idempotent
    logging.getLogger("tokenwatt.proxy").info("req.start", extra=event(model="m1"))
    for h in root.handlers:
        h.flush()
    lines = [l for l in open(f).read().splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["model"] == "m1"
```

- [ ] **Step 2: Run — expect FAIL** (`No module named tokenwatt.log`): `uv run pytest tests/test_log.py -q`

- [ ] **Step 3: Implement `src/tokenwatt/log.py`**

```python
from __future__ import annotations
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def event(**fields) -> dict:
    """Build the `extra=` payload for a structured log call:
    logger.info("req.start", extra=event(model=..., in_flight=...))."""
    return {"tw": fields}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event (the message), + the `tw` fields."""
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        tw = getattr(record, "tw", None)
        if isinstance(tw, dict):
            for k, v in tw.items():
                if k not in obj:
                    obj[k] = v
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


class ConsoleFormatter(logging.Formatter):
    """Concise human line for stderr: '<LEVEL> <event>  k=v k=v'."""
    def format(self, record: logging.LogRecord) -> str:
        tw = getattr(record, "tw", None)
        kv = " ".join(f"{k}={v}" for k, v in tw.items()) if isinstance(tw, dict) else ""
        return f"{record.levelname:7} {record.getMessage()}  {kv}".rstrip()


def setup_logging(*, level: str = "INFO", file: str | None = "~/.tokenwatt/logs/proxy.jsonl",
                  console: bool = True, max_bytes: int = 10_485_760, backup_count: int = 5) -> None:
    """Configure the `tokenwatt` logger tree. Idempotent: clears prior handlers first."""
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    tw_logger = logging.getLogger("tokenwatt")   # the tokenwatt PACKAGE logger, not the process root
    tw_logger.setLevel(lvl)
    tw_logger.propagate = False                  # stop tokenwatt.* double-emitting via root; uvicorn loggers untouched
    for h in list(tw_logger.handlers):
        tw_logger.removeHandler(h)
        h.close()
    if file:
        path = os.path.expanduser(file)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(JsonLineFormatter())
        tw_logger.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(ConsoleFormatter())
        tw_logger.addHandler(sh)
```

- [ ] **Step 4: Run — expect PASS**: `uv run pytest tests/test_log.py -q && uv run pytest -q`

- [ ] **Step 5: Commit** — `git add src/tokenwatt/log.py tests/test_log.py && git commit -m "feat(log): JSONL formatter + console formatter + setup_logging"`

---

### Task 2: `config.py` — `LoggingConfig`

**Files:** Modify `src/tokenwatt/config.py`; create `tests/test_config_logging.py`.

**Interfaces — Consumes:** none. **Produces:** `LoggingConfig` (pydantic) with `level/file/console/max_bytes/backup_count`; `Config.logging: LoggingConfig`.

- [ ] **Step 1: Failing test** (`tests/test_config_logging.py`)

```python
import tempfile, os
from tokenwatt.config import load_config, Config


def test_logging_defaults():
    cfg = Config()
    assert cfg.logging.level == "INFO"
    assert cfg.logging.file.endswith("proxy.jsonl")
    assert cfg.logging.console is True
    assert cfg.logging.max_bytes == 10_485_760 and cfg.logging.backup_count == 5


def test_logging_section_parsed(tmp_path):
    p = tmp_path / "tw.yaml"
    p.write_text("logging:\n  level: DEBUG\n  console: false\n  file: /tmp/x.jsonl\n")
    cfg = load_config(str(p))
    assert cfg.logging.level == "DEBUG" and cfg.logging.console is False and cfg.logging.file == "/tmp/x.jsonl"
```

- [ ] **Step 2: Run — expect FAIL**: `uv run pytest tests/test_config_logging.py -q`

- [ ] **Step 3: Implement** — in `src/tokenwatt/config.py`, add the model and the field:

```python
class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = "~/.tokenwatt/logs/proxy.jsonl"
    console: bool = True
    max_bytes: int = 10_485_760
    backup_count: int = 5
```

In `Config`, add (alongside `routes`):

```python
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

- [ ] **Step 4: Run — expect PASS**: `uv run pytest tests/test_config_logging.py -q && uv run pytest -q`

- [ ] **Step 5: Commit** — `git add src/tokenwatt/config.py tests/test_config_logging.py && git commit -m "feat(config): LoggingConfig section"`

---

### Task 3: `ledger.py` — `request_id` column

**Files:** Modify `src/tokenwatt/ledger.py`, `tests/test_ledger.py`.

**Interfaces — Produces:** `LedgerRow.request_id: str = ""`; `requests.request_id` column (migrated forward).

- [ ] **Step 1: Failing test** (append to `tests/test_ledger.py`)

```python
def test_request_id_column_roundtrips(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", request_id="abc123"))   # adapt to the real _row helper
    with led._conn() as c:
        assert c.execute("SELECT request_id FROM requests").fetchone()["request_id"] == "abc123"


def test_request_id_defaults_empty(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1"))
    with led._conn() as c:
        assert c.execute("SELECT request_id FROM requests").fetchone()["request_id"] == ""
```

(If the `_row` helper doesn't forward arbitrary kwargs, build `LedgerRow` directly mirroring an existing test. `request_id` defaults to `""`.)

- [ ] **Step 2: Run — expect FAIL**: `uv run pytest tests/test_ledger.py -q`

- [ ] **Step 3: Implement** — in `_SCHEMA`, append `request_id` to the requests columns:

```python
    req_type TEXT DEFAULT 'text', cold INTEGER DEFAULT 0, in_flight INTEGER DEFAULT 1,
    request_id TEXT DEFAULT ''
```

In `LedgerRow`, after `in_flight`:

```python
    in_flight: int = 1
    request_id: str = ""
```

In `_migrate`, after the `in_flight` block:

```python
        if "request_id" not in cols:
            c.execute("ALTER TABLE requests ADD COLUMN request_id TEXT DEFAULT ''")
```

- [ ] **Step 4: Run — expect PASS**: `uv run pytest tests/test_ledger.py -q && uv run pytest -q`

- [ ] **Step 5: Commit** — `git add src/tokenwatt/ledger.py tests/test_ledger.py && git commit -m "feat(ledger): request_id column for log correlation"`

---

### Task 4: `proxy.py` — lifecycle event logging

**Files:** Modify `src/tokenwatt/proxy.py`, `tests/test_proxy.py`.

**Interfaces — Consumes:** `tokenwatt.log.event` (Task 1), `LedgerRow.request_id` (Task 3). **Produces:** structured events on the `tokenwatt.proxy` logger; `request_id=label` on the ledger row.

- [ ] **Step 1: Failing tests** (append to `tests/test_proxy.py`)

```python
def _read_jsonl(path):
    import json
    return [json.loads(l) for l in open(path).read().splitlines() if l.strip()]


async def test_logs_start_and_finish_with_request_id(tmp_path, fake_upstream_json):
    from tokenwatt.log import setup_logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json),
                     detector=ColdStartDetector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw") as c:
        await c.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    import logging
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    events = _read_jsonl(logf)
    by = {e["event"]: e for e in events}
    assert "req.start" in by and "req.finish" in by
    rid = by["req.start"]["request_id"]
    assert rid and by["req.finish"]["request_id"] == rid            # shared correlation id
    with ledger._conn() as conn:
        assert conn.execute("SELECT request_id FROM requests").fetchone()["request_id"] == rid  # ledger <-> log


async def test_logs_stream_error_on_upstream_drop(tmp_path):
    # the upstream body read raises mid-stream -> a req.stream_error event with the error type
    from tokenwatt.log import setup_logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    app = create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=httpx.AsyncClient(transport=_BodyRaisingTransport()),
                     detector=ColdStartDetector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw", timeout=5.0) as c:
        try:
            await c.post("/v1/chat/completions",
                         json={"model": "m1", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        except Exception:
            pass
    import logging
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    errs = [e for e in _read_jsonl(logf) if e["event"] == "req.stream_error"]
    assert len(errs) == 1 and errs[0]["error_type"] == "ReadError"   # logged exactly once


# A streaming upstream that yields ONE frame then blocks, so the client can disconnect
# mid-stream and inject GeneratorExit into body_iter (the headline distinction).
class _OneFrameThenBlock(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        await asyncio.Event().wait()                  # block until the response is closed
    async def aclose(self):
        pass


class _GatedStreamTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        return httpx.Response(200, stream=_OneFrameThenBlock())


async def test_logs_client_disconnect_not_stream_error(tmp_path):
    # THE headline distinction: a client hangup logs req.client_disconnect (INFO), NOT
    # req.stream_error. Guards the except-clause ordering against a future reorder.
    from tokenwatt.log import setup_logging
    import asyncio, logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    app = create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=httpx.AsyncClient(transport=_GatedStreamTransport()),
                     detector=ColdStartDetector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw",
                                 timeout=5.0) as c:
        async with c.stream("POST", "/v1/chat/completions",
                            json={"model": "m1", "stream": True,
                                  "messages": [{"role": "user", "content": "hi"}]}) as resp:
            await resp.aiter_raw().__anext__()        # pull one chunk, then exit -> aclose mid-stream
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    kinds = {e["event"] for e in _read_jsonl(logf)}
    assert "req.client_disconnect" in kinds
    assert "req.stream_error" not in kinds            # a disconnect is NOT an error
    cd = next(e for e in _read_jsonl(logf) if e["event"] == "req.client_disconnect")
    assert cd["level"] == "INFO"
    # contract: req.finish still fires (row booked) but is stamped aborted="client_disconnect"
    fin = [e for e in _read_jsonl(logf) if e["event"] == "req.finish"]
    assert fin and fin[0]["aborted"] == "client_disconnect"


async def test_logs_no_route(tmp_path):
    from tokenwatt.log import setup_logging
    import logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    router = Router([RouteConfig(name="only", type="text", upstream="http://up", match=["specific-model"])])
    app = create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=httpx.AsyncClient(), detector=ColdStartDetector())   # client unused on 404
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw") as c:
        assert (await c.post("/v1/chat/completions", json={"model": "nope", "messages": []})).status_code == 404
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    nr = [e for e in _read_jsonl(logf) if e["event"] == "req.no_route"]
    assert nr and nr[0]["level"] == "WARNING" and nr[0]["model"] == "nope"


async def test_logs_upstream_error(tmp_path):
    from tokenwatt.log import setup_logging
    import logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    app = create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=httpx.AsyncClient(transport=_RaisingTransport()), detector=ColdStartDetector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw") as c:
        assert (await c.post("/v1/chat/completions", json={"model": "m1", "messages": []})).status_code == 502
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    ue = [e for e in _read_jsonl(logf) if e["event"] == "req.upstream_error"]
    assert ue and ue[0]["level"] == "ERROR"
```

(`_RaisingTransport`, `_BodyRaisingTransport`/`_RaisingStream`, `_client_for`, `fake_upstream_json`, `_ROUTER`, `Router`, `RouteConfig` all already exist at the top of `tests/test_proxy.py`. Also extend the existing `test_nonstream_finalize_runs_on_midbody_error_no_leak` to assert a `req.stream_error` line is emitted on the NON-stream path too.)

- [ ] **Step 2: Run — expect FAIL**: `uv run pytest tests/test_proxy.py -q`

- [ ] **Step 3: Implement** — in `src/tokenwatt/proxy.py`:

(a) Top of file, add imports:

```python
import asyncio
import logging
...
from tokenwatt.log import event

logger = logging.getLogger("tokenwatt.proxy")
```

(b) The 404 path — before `return JSONResponse(... 404 ...)` add:

```python
            logger.warning("req.no_route", extra=event(model=model, routes=router.route_names()))
```

(c) After `in_flight = idle.in_flight()` (and t0/ts_start), add the start event:

```python
        logger.info("req.start", extra=event(request_id=label, model=ledger_model, req_type=req_type,
                                             stream=is_stream, upstream=route.upstream,
                                             in_flight=in_flight, body_bytes=len(raw)))
```

(d) The upstream-send failure — bind the exception and log:

```python
        except Exception as e:
            meter.end(label)
            idle.request_finished()
            if serialize_lock is not None:
                serialize_lock.release()
            logger.error("req.upstream_error", extra=event(request_id=label, model=ledger_model,
                                                           error_type=type(e).__name__, error=str(e)[:300]))
            return JSONResponse(
                {"error": {"message": "upstream request failed", "type": "upstream_error"}},
                status_code=502,
            )
```

(e) Declare an abort marker `aborted: list = []` immediately after the existing `first_chunk_t: list[float] = []` line (closure-shared, like `first_chunk_t`). Then in `_finalize`, pass `request_id=label` to the `LedgerRow(...)` (add `request_id=label,` next to `in_flight=in_flight,`), and after the `ledger.insert(...)` call (still inside the `try`), add the finish event. `aborted` makes `req.finish` self-describing — `None` on success, else the abort reason — so `req.finish` is NOT itself a "success" signal; an aborted stream emits both the abort event AND `req.finish` for the same `request_id`, and the `aborted` field disambiguates:

```python
                logger.info("req.finish", extra=event(
                    request_id=label, model=ledger_model, duration_s=round(dt, 2),
                    ttft_s=round(ttft, 2) if ttft is not None else None, status=up_resp.status_code,
                    tok_in=usage.input if usage else None, tok_out=usage.output if usage else None,
                    tok_source=usage.source if usage else "none", marginal_j=round(marginal_j, 1),
                    kwh=kwh, cost=cost, cold=cold.is_cold,
                    aborted=(aborted[0] if aborted else None)))
```

(f) The streaming `body_iter` — count chunks/bytes and capture stream errors. Replace it with:

```python
        if is_stream:
            async def body_iter():
                n = 0; nb = 0
                try:
                    async for chunk in up_resp.aiter_raw():
                        if not first_chunk_t:
                            first_chunk_t.append(time.monotonic())
                        n += 1; nb += len(chunk)
                        counter.feed(chunk)
                        yield chunk
                except (GeneratorExit, asyncio.CancelledError):
                    aborted.append("client_disconnect")
                    logger.info("req.client_disconnect", extra=event(
                        request_id=label, model=ledger_model, chunks=n, bytes=nb,
                        elapsed_s=round(time.monotonic() - t0, 2)))
                    raise
                except Exception as e:
                    aborted.append("stream_error")
                    logger.warning("req.stream_error", extra=event(
                        request_id=label, model=ledger_model, error_type=type(e).__name__,
                        error=str(e)[:300], chunks=n, bytes=nb,
                        elapsed_s=round(time.monotonic() - t0, 2)))
                    raise
                finally:
                    await up_resp.aclose()
                    _finalize(counter.result())
            return StreamingResponse(body_iter(), status_code=up_resp.status_code,
                                     headers=dict(resp_headers))
```

(g) The non-streaming branch — log a stream_error if the body read raises (preserve the existing finalize + propagation):

```python
        else:
            n = 0; nb = 0
            try:
                async for chunk in up_resp.aiter_raw():
                    n += 1; nb += len(chunk)
                    captured.extend(chunk)
            except Exception as e:
                aborted.append("stream_error")
                logger.warning("req.stream_error", extra=event(
                    request_id=label, model=ledger_model, error_type=type(e).__name__,
                    error=str(e)[:300], chunks=n, bytes=nb, elapsed_s=round(time.monotonic() - t0, 2)))
                raise
            finally:
                await up_resp.aclose()
                body = bytes(captured)
                try:
                    usage = usage_from_response_json(json.loads(body))
                except json.JSONDecodeError:
                    usage = None
                _finalize(usage or TokenUsage(None, None, None, "none", "energy-only"))
            return Response(content=body, status_code=up_resp.status_code,
                            headers=dict(resp_headers))
```

- [ ] **Step 4: Run — expect PASS** (existing proxy tests still pass — events are additive): `uv run pytest tests/test_proxy.py -q && uv run pytest -q`

- [ ] **Step 5: Commit** — `git add src/tokenwatt/proxy.py tests/test_proxy.py && git commit -m "feat(proxy): structured lifecycle logging + stream-error capture + request_id"`

---

### Task 5: `cli.py serve` — wire setup_logging with precedence

**Files:** Modify `src/tokenwatt/cli.py`; add a precedence unit test (`tests/test_cli_config.py` or a new file).

**Interfaces — Consumes:** `setup_logging` (Task 1), `Config.logging` (Task 2). **Produces:** `serve` configures logging from CLI > env > config; helper `_effective_log(cli_level, cli_file, cfg_logging) -> (level, file)`.

- [ ] **Step 1: Failing test** (a unit test for precedence; append to `tests/test_cli_config.py` or new `tests/test_cli_logging.py`)

```python
import os
from tokenwatt.cli import _effective_log
from tokenwatt.config import LoggingConfig


def test_log_precedence_cli_over_env_over_config(monkeypatch):
    cfg = LoggingConfig(level="WARNING", file="/c.jsonl")
    monkeypatch.setenv("TOKENWATT_LOG_LEVEL", "INFO")
    # CLI wins
    assert _effective_log("DEBUG", "/cli.jsonl", cfg) == ("DEBUG", "/cli.jsonl")
    # env wins over config when no CLI level
    assert _effective_log(None, None, cfg)[0] == "INFO"
    # config file used when no CLI file
    assert _effective_log(None, None, cfg)[1] == "/c.jsonl"


def test_log_precedence_config_when_no_cli_or_env(monkeypatch):
    monkeypatch.delenv("TOKENWATT_LOG_LEVEL", raising=False)
    cfg = LoggingConfig(level="ERROR", file="/c.jsonl")
    assert _effective_log(None, None, cfg) == ("ERROR", "/c.jsonl")
    monkeypatch.setenv("TOKENWATT_LOG_LEVEL", "")          # empty env is treated as unset
    assert _effective_log(None, None, cfg)[0] == "ERROR"
```

- [ ] **Step 2: Run — expect FAIL** (`_effective_log` undefined): `uv run pytest tests/test_cli_logging.py -q`

- [ ] **Step 3: Implement** — in `src/tokenwatt/cli.py`:

Add the helper near the top (after imports):

```python
def _effective_log(cli_level, cli_file, cfg_logging):
    import os
    # precedence: CLI > env > config. Spec: only LEVEL has an env override
    # (TOKENWATT_LOG_LEVEL), not file. An empty/unset env falls through to config.
    level = cli_level or os.environ.get("TOKENWATT_LOG_LEVEL") or cfg_logging.level
    file = cli_file or cfg_logging.file
    return level, file
```

Add two options to the `serve` command signature:

```python
    log_level: Optional[str] = typer.Option(None, "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
```

In `serve`, after `cfg = load_config(config)` (and the upstream/rate overrides), before building the app:

```python
    from tokenwatt.log import setup_logging
    _lvl, _file = _effective_log(log_level, log_file, cfg.logging)
    setup_logging(level=_lvl, file=_file, console=cfg.logging.console,
                  max_bytes=cfg.logging.max_bytes, backup_count=cfg.logging.backup_count)
```

- [ ] **Step 4: Run — expect PASS**: `uv run pytest tests/test_cli_logging.py -q && uv run pytest -q`

- [ ] **Step 5: Commit** — `git add src/tokenwatt/cli.py tests/test_cli_logging.py && git commit -m "feat(cli): serve configures structured logging (CLI>env>config)"`

---

## Self-Review

**1. Spec coverage:**
- Module-scoped named loggers → Task 4 (`tokenwatt.proxy`); the tree configured in Task 1. ✓
- JSONL + rotating file + console → Task 1 (`setup_logging`, both handlers). ✓
- `req.start`/`req.finish`/`req.stream_error`/`req.client_disconnect`/`req.upstream_error`/`req.no_route` → Task 4. ✓
- Stream-error distinguishes upstream vs client disconnect → Task 4 (f): `except (GeneratorExit, asyncio.CancelledError)` (client) before `except Exception` (upstream). ✓
- Privacy (no content) → only sizes/counts logged; verify no message field carries body. ✓
- request_id correlation (log + ledger) → Task 3 (column) + Task 4 (e: LedgerRow + start/finish events share `label`). ✓
- Config + CLI + env precedence → Task 2 + Task 5. ✓
- Defaults (INFO, ~/.tokenwatt/logs/proxy.jsonl, 10MB×5, console on) → Task 1 + Task 2. ✓
- Idempotent setup → Task 1 (clears handlers). ✓
- Eventually-bucket (/metrics, OTel, alerting, async logging) correctly NOT built. ✓

**2. Placeholder scan:** none — every step has complete code + expected output.

**3. Type consistency:** `event(**fields)->{"tw":fields}` consumed by `JsonLineFormatter` (reads `record.tw`) and by every proxy call. `setup_logging(level,file,console,max_bytes,backup_count)` signature matches the `serve` call in Task 5. `_effective_log(cli_level,cli_file,cfg_logging)->(level,file)` matches its test + caller. `LedgerRow.request_id` (Task 3) set in Task 4. `up_resp.status_code` read in `_finalize` (Task 4e) — `up_resp` is assigned before `_finalize` is defined, so it is in closure scope. ✓

**Known limitation (stated):** v1 logging is synchronous (file writes on the event-loop thread at lifecycle boundaries only — 2 lines/request, off the per-chunk path — so negligible for local-inference QPS). A `QueueHandler`/`QueueListener` async path is in the eventually-bucket for higher throughput.
