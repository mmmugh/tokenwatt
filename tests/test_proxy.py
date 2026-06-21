import asyncio
import json
import httpx
import pytest

from tokenwatt.proxy import create_app
from tokenwatt.meter import EnergyByRail, FakeMeter
from tokenwatt.idle import IdleBaseline
from tokenwatt.rate import FlatRate
from tokenwatt.ledger import Ledger
from tokenwatt.config import RouteConfig
from tokenwatt.router import Router
from tokenwatt.coldstart import ColdStartDetector

_ROUTER = Router([RouteConfig(name="m1", type="text", upstream="http://up", match=["*"])])


class _RaisingTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise httpx.ConnectError("boom")


def _client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://up")


async def test_streaming_passthrough_is_byte_exact_and_records(tmp_path, fake_upstream_streaming):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    meter = FakeMeter(windows={"x": EnergyByRail({"gpu": 3_600_000.0})})  # 1 kWh window
    # Force the request id label so FakeMeter returns our window:
    app = create_app(
        router=_ROUTER, meter=meter, idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(0.31),
        client=_client_for(fake_upstream_streaming),
        detector=ColdStartDetector(),
        _label_factory=lambda: "x",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        body = {"model": "m1", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        r = await c.post("/v1/chat/completions", json=body)
        assert r.status_code == 200
        # byte-exact: the three SSE frames survive untouched
        assert r.text == (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            'data: [DONE]\n\n'
        )
    rows = ledger.by_model()
    assert rows[0]["model"] == "m1"
    assert abs(rows[0]["total_usd"] - 0.31) < 1e-9   # 1 kWh * $0.31


async def test_json_passthrough_uses_backend_usage(tmp_path, fake_upstream_json):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(
        router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(0.31),
        client=_client_for(fake_upstream_json),
        detector=ColdStartDetector(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        assert r.json()["usage"]["completion_tokens"] == 2
    with ledger._conn() as conn:
        row = conn.execute("SELECT tok_out, tok_source FROM requests").fetchone()
    assert row["tok_out"] == 2 and row["tok_source"] == "backend"


async def test_upstream_error_returns_502_and_closes_window(tmp_path):
    # When the upstream send raises, the proxy must return 502 and must not leak
    # the open energy window (meter._open must be empty after the call).
    meter = FakeMeter()
    idle = IdleBaseline(FakeMeter())
    failing_client = httpx.AsyncClient(transport=_RaisingTransport())
    app = create_app(
        router=_ROUTER, meter=meter, idle=idle,
        ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
        client=failing_client,
        detector=ColdStartDetector(),
        _label_factory=lambda: "x",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert meter._open == set()   # no leaked window


async def test_cost_none_when_rate_unset(tmp_path, fake_upstream_json):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(
        router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(None),
        client=_client_for(fake_upstream_json),
        detector=ColdStartDetector(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    with ledger._conn() as conn:
        row = conn.execute("SELECT cost_marginal_usd FROM requests").fetchone()
    assert row["cost_marginal_usd"] is None


async def test_streaming_prefers_backend_usage_chunk(tmp_path, fake_upstream_streaming_with_usage):
    # A streamed response carrying a usage chunk must be recorded from the backend's
    # numbers (source=backend, tok_out=9), not self-counted from content deltas.
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(
        router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(0.31),
        client=_client_for(fake_upstream_streaming_with_usage),
        detector=ColdStartDetector(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "m1", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]})
    with ledger._conn() as conn:
        row = conn.execute("SELECT tok_out, tok_source FROM requests").fetchone()
    assert row["tok_out"] == 9 and row["tok_source"] == "backend"


async def test_unmatched_model_returns_404(tmp_path, fake_upstream_json):
    router = Router([RouteConfig(name="m1", type="text", upstream="http://up", match=["only-m1"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    meter = FakeMeter()
    app = create_app(router=router, meter=meter, idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json),
                     detector=ColdStartDetector())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "something-else", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert "m1" in r.json()["error"]["message"]   # 404 lists the configured route name
    assert meter._open == set()   # a 404 must NOT open/leak an energy window
    with ledger._conn() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"] == 0


async def test_ledger_records_requested_model_id_for_namespaced_hf(tmp_path, fake_upstream_json):
    # client sends a namespaced HF id; the route named "v1" matches it by glob ->
    # ledger records the actual requested model id so each model gets its own row.
    router = Router([RouteConfig(name="v1", type="vision", upstream="http://up", match=["*VL*"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json),
                     detector=ColdStartDetector())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "mlx-community/Qwen3-VL-8B", "messages": [{"role": "user", "content": "hi"}]})
    assert ledger.by_model()[0]["model"] == "mlx-community/Qwen3-VL-8B"


async def test_vision_request_records_req_type(tmp_path, fake_upstream_json):
    router = Router([RouteConfig(name="v1", type="vision", upstream="http://up", match=["*"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json),
                     detector=ColdStartDetector())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions", json={"model": "v1", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]})
    with ledger._conn() as conn:
        assert conn.execute("SELECT req_type FROM requests").fetchone()["req_type"] == "vision"


async def test_embeddings_request_records_req_type(tmp_path, fake_upstream_json):
    router = Router([RouteConfig(name="embed", type="embeddings", upstream="http://up", match=["*"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json),
                     detector=ColdStartDetector())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/embeddings", json={"model": "embed", "input": "hello world"})
    with ledger._conn() as conn:
        assert conn.execute("SELECT req_type FROM requests").fetchone()["req_type"] == "embedding"


async def test_cold_start_books_model_load_and_flags_request(tmp_path, fake_upstream_slow_first_chunk):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    # Pre-seed a fast warm baseline for "m1" so the proxy request's ~0.4s TTFT (the fixture's
    # load delay) lands ABOVE threshold and is detected cold (the FIRST observation only seeds).
    detector = ColdStartDetector(floor_s=0.1, factor=2.0)
    for _ in range(3):
        detector.observe("m1", 0.02, 100.0, 1.0)              # warm baseline ~0.02s; threshold max(0.1, 0.04)=0.1
    meter = FakeMeter(windows={"x": EnergyByRail({"gpu": 1000.0})})
    app = create_app(router=Router([RouteConfig(name="m1", type="text", upstream="http://up", match=["*"])]),
                     meter=meter, idle=IdleBaseline(FakeMeter()), ledger=ledger, rate=FlatRate(0.31),
                     client=_client_for(fake_upstream_slow_first_chunk), detector=detector,
                     _label_factory=lambda: "x")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "m1", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert ledger.model_load_summary()["count"] == 1          # a model_load row was booked
    with ledger._conn() as conn:
        ml = conn.execute("SELECT load_energy_j, duration_ms FROM model_loads").fetchone()
        row = conn.execute("SELECT e_window_j, e_idle_j, e_marginal_j, cold FROM requests").fetchone()
    assert row["cold"] == 1                                    # request flagged cold
    assert ml["load_energy_j"] > 100.0                         # a meaningful load chunk was booked
    assert ml["duration_ms"] > 100.0                           # ~0.4s load duration persisted (ms)
    assert 0.0 <= row["e_marginal_j"] < 900.0                  # load subtracted (not the trivial < 1000)
    # energy conservation: full window = idle + booked load + inference marginal
    assert abs(row["e_window_j"] - (row["e_idle_j"] + ml["load_energy_j"] + row["e_marginal_j"])) < 1e-6


async def test_non_stream_request_is_never_cold(tmp_path, fake_upstream_json):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=Router([RouteConfig(name="m1", type="text", upstream="http://up", match=["*"])]),
                     meter=FakeMeter(), idle=IdleBaseline(FakeMeter()), ledger=ledger, rate=FlatRate(0.31),
                     client=_client_for(fake_upstream_json), detector=ColdStartDetector())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions", json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    assert ledger.model_load_summary()["count"] == 0
    with ledger._conn() as conn:
        assert conn.execute("SELECT cold FROM requests").fetchone()["cold"] == 0


class _JsonStream(httpx.AsyncByteStream):
    """Single-use async stream that yields one JSON payload then stops."""
    def __init__(self, body: bytes):
        self._body = body

    async def __aiter__(self):
        yield self._body

    async def aclose(self):
        pass


class _GatedTransport(httpx.AsyncBaseTransport):
    """Upstream that blocks until `release` is set (to force overlap).

    Uses a fresh AsyncByteStream per call so httpx doesn't pre-consume the body
    and raise StreamConsumed on the second concurrent response (httpx 0.28.x).
    """
    def __init__(self, release: asyncio.Event):
        self._release = release

    async def handle_async_request(self, request):
        await self._release.wait()
        import json as _json
        body = _json.dumps({"choices": [{"message": {"content": "ok"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        return httpx.Response(200, stream=_JsonStream(body),
                              headers={"content-type": "application/json"})


class _RaisingStream(httpx.AsyncByteStream):
    """A response body whose first read raises (simulates a mid-body upstream drop)."""
    async def __aiter__(self):
        if False:
            yield b""            # make it an async generator
        raise httpx.ReadError("mid-body boom")

    async def aclose(self):
        pass


class _BodyRaisingTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        return httpx.Response(200, stream=_RaisingStream())


def _proxy_app(ledger, *, serialize_lock, client):
    return create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                      ledger=ledger, rate=FlatRate(0.31), client=client,
                      detector=ColdStartDetector(), serialize_lock=serialize_lock)


async def test_nonstream_finalize_runs_on_midbody_error_no_leak(tmp_path):
    # the non-stream body read raises; the proxy must still release in_flight + close the window
    # AND emit a req.stream_error log event
    from tokenwatt.log import setup_logging
    import logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    meter = FakeMeter(); idle = IdleBaseline(FakeMeter())
    app = create_app(router=_ROUTER, meter=meter, idle=idle,
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=httpx.AsyncClient(transport=_BodyRaisingTransport()),
                     detector=ColdStartDetector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw",
                                 timeout=5.0) as c:
        try:
            await c.post("/v1/chat/completions",
                         json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        except Exception:
            pass                                   # a 500 / propagated error is acceptable
    assert idle._in_flight == 0                    # in-flight released despite the error
    assert meter._open == set()                    # the energy window was closed
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    errs = [e for e in _read_jsonl(logf) if e["event"] == "req.stream_error"]
    assert len(errs) == 1 and errs[0]["error_type"] == "ReadError"  # non-stream path also logs


async def test_in_flight_detects_overlap_without_lock(tmp_path):
    release = asyncio.Event()
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = _proxy_app(ledger, serialize_lock=None,
                     client=httpx.AsyncClient(transport=_GatedTransport(release)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw",
                                 timeout=5.0) as c:
        async def fire():
            return await c.post("/v1/chat/completions",
                                json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        t1 = asyncio.create_task(fire()); t2 = asyncio.create_task(fire())
        await asyncio.sleep(0.05)                  # both reach the gated upstream -> both in flight
        release.set()
        await asyncio.gather(t1, t2)
    with ledger._conn() as c:
        flights = sorted(r["in_flight"] for r in c.execute("SELECT in_flight FROM requests"))
    assert flights[-1] == 2                         # overlap is detected in the audit column


async def test_ledger_records_requested_model_not_route_name(tmp_path, fake_upstream_json):
    # route is a catch-all named "m1"; the REQUEST asks for a different model id ->
    # the ledger must record the requested id, so LM Studio's many models show separately
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(
        router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(0.31),
        client=_client_for(fake_upstream_json),
        detector=ColdStartDetector(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "lmstudio-community/Qwen3-Coder-30B", "messages": [{"role": "user", "content": "hi"}]})
    with ledger._conn() as conn:
        assert conn.execute("SELECT model FROM requests").fetchone()["model"] == "lmstudio-community/Qwen3-Coder-30B"


async def test_serialize_lock_prevents_overlap(tmp_path):
    release = asyncio.Event(); release.set()        # upstream returns immediately
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = _proxy_app(ledger, serialize_lock=asyncio.Lock(),
                     client=httpx.AsyncClient(transport=_GatedTransport(release)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw",
                                 timeout=5.0) as c:
        async def fire():
            return await c.post("/v1/chat/completions",
                                json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        await asyncio.gather(*[asyncio.create_task(fire()) for _ in range(4)])
    with ledger._conn() as c:
        flights = [r["in_flight"] for r in c.execute("SELECT in_flight FROM requests")]
    assert flights == [1, 1, 1, 1]                  # the lock serialized -> never two windows open


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


# Upstream that blocks after the first chunk so we can inject a simulated disconnect
# without the response completing. _OneFrameThenBlock is used to provide a hanging
# upstream; the disconnect is simulated via a direct ASGI call (see test below).
class _OneFrameThenBlock(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        await asyncio.Event().wait()                  # blocks until aclose() cancels it
    async def aclose(self):
        pass


class _GatedStreamTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        return httpx.Response(200, stream=_OneFrameThenBlock())


async def test_logs_client_disconnect_not_stream_error(tmp_path):
    # THE headline distinction: a client hangup logs req.client_disconnect (INFO), NOT
    # req.stream_error. Guards the except-clause ordering against a future reorder.
    #
    # Fixture adaptation: httpx.ASGITransport buffers the full response body before
    # returning, so a mid-stream close via httpx.stream() never reaches the ASGI
    # receive() callable as http.disconnect (the transport blocks waiting for
    # response_complete). We therefore drive the ASGI app directly: run it as a task,
    # inject http.disconnect via a shared event, and cancel the task once we have the
    # first chunk — replicating exactly what Starlette's anyio task group does when a
    # real client hangs up. This is the mechanism that injects CancelledError into
    # body_iter; CancelledError is caught by the (GeneratorExit, CancelledError) clause
    # and logged as req.client_disconnect.
    from tokenwatt.log import setup_logging
    import logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    app = create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=httpx.AsyncClient(transport=_GatedStreamTransport()),
                     detector=ColdStartDetector())

    # Build a minimal ASGI scope for a streaming POST
    import json as _json
    body = _json.dumps({"model": "m1", "stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 9999),
        "root_path": "",
    }

    got_first_chunk = asyncio.Event()
    disconnect_ready = asyncio.Event()
    chunks = []

    async def receive():
        # First call: supply the request body. Subsequent calls: simulate disconnect.
        if not disconnect_ready.is_set():
            disconnect_ready.set()
            return {"type": "http.request", "body": body, "more_body": False}
        # Block until we've seen one response chunk, then signal disconnect
        await got_first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if chunk:
                chunks.append(chunk)
                got_first_chunk.set()     # unblock receive() -> http.disconnect

    # Run the ASGI app as a task and cancel it after the disconnect fires
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(got_first_chunk.wait(), timeout=5.0)
        # Wait (bounded) for the disconnect to propagate into body_iter and be logged,
        # instead of a fixed sleep that can flake on a loaded runner.
        for _ in range(100):                          # up to ~5s
            for h in logging.getLogger("tokenwatt").handlers:
                h.flush()
            if any(e["event"] == "req.client_disconnect" for e in _read_jsonl(logf)):
                break
            await asyncio.sleep(0.05)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

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


async def test_no_prompt_content_in_logs(tmp_path, fake_upstream_json):
    # privacy invariant: prompt/response CONTENT must never appear in the structured log,
    # only sizes/counts. Sentinel guards against a future event that logs body/messages.
    from tokenwatt.log import setup_logging
    import logging
    logf = str(tmp_path / "p.jsonl")
    setup_logging(level="INFO", file=logf, console=False)
    SENTINEL = "XYZZY_secret_prompt_PLUGH"
    app = create_app(router=_ROUTER, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                     client=_client_for(fake_upstream_json), detector=ColdStartDetector())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "m1", "messages": [{"role": "user", "content": SENTINEL}]})
    for h in logging.getLogger("tokenwatt").handlers:
        h.flush()
    assert SENTINEL not in open(logf).read()
