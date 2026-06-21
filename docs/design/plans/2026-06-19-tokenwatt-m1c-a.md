# TokenWatt M1c-a — Cold-Start / Model-Load Booking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a request pays one-time model-load energy (LM Studio / load-on-demand backends swapping a model in), detect it via time-to-first-token, book the load energy to a separate `model_load` row, and keep the request's per-token cost clean — so a cold first hit doesn't look 5–50× too expensive.

**Architecture:** The proxy already tees the streamed response; it records **time-to-first-token (TTFT)** = first chunk arrival − request start. A stateful `ColdStartDetector` keeps a per-model rolling-median *warm* TTFT; a request whose TTFT far exceeds that (or a fixed floor when no baseline exists) is **cold**. The detector estimates the load's time (`TTFT − warm`) and apportions the window's marginal energy by that time fraction. The proxy subtracts the load energy from the request row, flags it `cold`, and writes a `model_load` row. Non-streaming requests (no TTFT) are left unflagged in v1.

**Tech Stack:** Python 3.12, existing stack (no new deps; `statistics.median`, `collections.deque`).

## Global Constraints

- **Python ≥ 3.12.** No new dependencies.
- **Cold detection is TTFT-based** (streaming only): the FIRST request for a model SEEDS its warm baseline (never cold); afterwards a request is cold when `ttft > max(floor_s, warm_baseline_median × factor)`. Defaults: `floor_s=1.0`, `factor=3.0`, baseline `window=16`. (A genuine first-ever load is missed — booked as inference — to avoid permanently mis-flagging a model whose warm TTFT exceeds the floor.)
- **Energy split:** `load_energy_j = min(1.0, max(0, ttft − baseline) / duration_s) × window_marginal_j`. The request row records the **inference** marginal (window − idle − load); the `model_load` row records the load energy AND its duration. This is a TIME-fraction ESTIMATE — it assumes uniform power across the window, so it OVER-books when the load phase draws less power than decode (spec §14: an estimate, not a measured value).
- **Cold requests are flagged** (`cold=True`) even after the split, so they're identifiable. A cold request's TTFT is NOT added to the warm baseline.
- **Non-streaming requests** (no TTFT) are NOT flagged cold in v1 (can't measure the split honestly) — deferred.
- **Backward-compatible ledger:** add a `cold` column (forward-only migration) and a new `model_loads` table; pre-M1c DBs keep working.
- **Byte-exact passthrough + routing + req_type (M0/M1a/M1b) are unchanged.** `create_app` gains a `detector` parameter (stateful across requests, like `idle`).
- **No sudo at runtime.** Version from `VERSION` (auto-bumped per commit). Conventional Commit prefixes.

---

## File Structure

```
src/tokenwatt/coldstart.py   # ColdStartDetector + ColdResult (TTFT-based detection + energy split)
src/tokenwatt/ledger.py      # MODIFIED: LedgerRow gains cold; migration; model_loads table + insert_model_load + model_load_summary
src/tokenwatt/proxy.py       # MODIFIED: measure TTFT (streaming), wire detector into _finalize, split energy + flag + model_load row; create_app gains detector=
src/tokenwatt/cli.py         # MODIFIED: serve builds a ColdStartDetector; render_report shows the model-load summary
tests/test_coldstart.py
tests/test_ledger.py         # MODIFIED: add cold column / model_loads tests
tests/test_proxy.py          # MODIFIED: add a cold-start integration test (slow-first-chunk fake upstream)
tests/test_report_render.py  # MODIFIED: model-load summary line
```

---

### Task 1: ColdStartDetector

**Files:**
- Create: `src/tokenwatt/coldstart.py`
- Test: `tests/test_coldstart.py`

**Interfaces:**
- Produces:
  - `ColdResult` dataclass: `is_cold: bool`, `load_energy_j: float`, `trigger: str` (`"transition"` | `"ttft_outlier"` | `"none"`).
  - `ColdStartDetector(window=16, floor_s=1.0, factor=3.0)` with `observe(model: str, ttft_s: float | None, window_marginal_j: float, duration_s: float) -> ColdResult(is_cold, load_energy_j, load_time_s, trigger)`. First sight of a model SEEDS its baseline (never cold); afterwards cold when `ttft_s > max(floor_s, baseline_median × factor)`. Warm TTFTs feed the per-model baseline; cold TTFTs do not. `ttft_s is None` → never cold, no split.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coldstart.py
from tokenwatt.coldstart import ColdStartDetector


def test_first_observation_seeds_baseline_and_is_not_cold():
    # First sight of a model has no warm baseline -> it SEEDS one and is NOT flagged cold
    # (avoids permanently mis-flagging a naturally-slow warm model).
    d = ColdStartDetector()
    r = d.observe("m1", ttft_s=5.0, window_marginal_j=600.0, duration_s=6.0)
    assert r.is_cold is False and r.load_energy_j == 0.0


def test_warm_request_is_not_cold():
    d = ColdStartDetector()
    assert d.observe("m1", 0.3, 100.0, 2.0).is_cold is False


def test_cold_books_energy_by_time_fraction():
    # Warm baseline ~0.2s, then a 5s TTFT on a 6s request -> load (5-0.2)/6 of 600 J = 480 J.
    d = ColdStartDetector(floor_s=0.5)               # baseline*3 (0.6) governs over the floor
    for _ in range(3):
        d.observe("m1", 0.2, 100.0, 2.0)             # baseline median 0.2
    r = d.observe("m1", ttft_s=5.0, window_marginal_j=600.0, duration_s=6.0)
    assert r.is_cold is True
    assert abs(r.load_energy_j - (4.8 / 6.0) * 600.0) < 1e-6   # 480 J
    assert abs(r.load_time_s - 4.8) < 1e-9
    assert r.trigger == "ttft_outlier"               # same model -> not a transition


def test_reload_uses_baseline_threshold_not_just_floor():
    # warm baseline 0.5 -> threshold = max(floor 1.0, 0.5*3=1.5) = 1.5, so the BASELINE governs.
    d = ColdStartDetector()
    for _ in range(4):
        d.observe("m1", 0.5, 100.0, 2.0)
    assert d.observe("m1", 1.2, 100.0, 2.0).is_cold is False   # below 1.5 -> warm (would be cold under floor-only)
    assert d.observe("m1", 4.0, 500.0, 5.0).is_cold is True    # above 1.5 -> cold


def test_non_streaming_none_ttft_is_never_cold():
    d = ColdStartDetector()
    r = d.observe("m1", ttft_s=None, window_marginal_j=999.0, duration_s=10.0)
    assert r.is_cold is False and r.load_energy_j == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_coldstart.py -q`
Expected: FAIL — `No module named tokenwatt.coldstart`.

- [ ] **Step 3: Implement `coldstart.py`**

```python
# src/tokenwatt/coldstart.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass
class ColdResult:
    is_cold: bool
    load_energy_j: float   # TIME-fraction ESTIMATE: assumes uniform power across the window, so
                           # it OVER-books when the load phase draws less power than decode.
                           # An estimate, not a measured load energy.
    load_time_s: float     # estimated load duration (ttft - warm baseline)
    trigger: str           # "transition" | "ttft_outlier" | "none"


class ColdStartDetector:
    """Detects model-load (cold-start) events from time-to-first-token and estimates the
    load energy. A load takes seconds; warm prefill is sub-second.

    The FIRST observation of a model SEEDS its warm baseline and is never flagged cold —
    this misses a genuine first-ever load (booked as inference) but is far safer than
    permanently mis-flagging a model whose warm TTFT naturally exceeds floor_s. Cold TTFTs
    do NOT feed the baseline."""

    def __init__(self, window: int = 16, floor_s: float = 1.0, factor: float = 3.0) -> None:
        self._last_model: str | None = None
        self._ttft: dict[str, deque[float]] = {}
        self._window = window
        self._floor_s = floor_s
        self._factor = factor

    def observe(self, model: str, ttft_s: float | None,
                window_marginal_j: float, duration_s: float) -> ColdResult:
        transition = model != self._last_model
        self._last_model = model

        if ttft_s is None:                       # non-streaming: can't measure a split
            return ColdResult(False, 0.0, 0.0, "none")

        samples = self._ttft.get(model)
        baseline = median(samples) if samples else None
        if baseline is None:                     # no warm baseline yet: SEED it, never cold
            self._ttft.setdefault(model, deque(maxlen=self._window)).append(ttft_s)
            return ColdResult(False, 0.0, 0.0, "none")

        threshold = max(self._floor_s, baseline * self._factor)
        if ttft_s <= threshold:                  # warm: feed the baseline, not cold
            self._ttft[model].append(ttft_s)
            return ColdResult(False, 0.0, 0.0, "none")

        load_time = max(0.0, ttft_s - baseline)
        frac = min(1.0, load_time / duration_s) if duration_s > 0 else 0.0
        return ColdResult(True, frac * window_marginal_j, load_time,
                          "transition" if transition else "ttft_outlier")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_coldstart.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/coldstart.py tests/test_coldstart.py
git commit -m "feat(coldstart): TTFT-based model-load detector with energy split"
```

---

### Task 2: Ledger — `cold` flag + `model_loads` table

**Files:**
- Modify: `src/tokenwatt/ledger.py`
- Modify: `tests/test_ledger.py`

**Interfaces:**
- Produces: `LedgerRow` gains `cold: bool = False` (last field). `Ledger.__init__` migrates: adds the `cold` column to `requests` if missing, and creates a `model_loads` table. New methods: `insert_model_load(ts, model, upstream, load_energy_j, duration_ms, trigger) -> None`; `model_load_summary() -> dict` returning `{"count": int, "total_load_j": float}`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_ledger.py`:

```python
def test_model_load_summary(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert_model_load(ts=100.0, model="m1", upstream="http://a", load_energy_j=480.0, duration_ms=4800.0, trigger="transition")
    led.insert_model_load(ts=101.0, model="v1", upstream="http://b", load_energy_j=120.0, duration_ms=1200.0, trigger="ttft_outlier")
    s = led.model_load_summary()
    assert s["count"] == 2 and abs(s["total_load_j"] - 600.0) < 1e-9


def test_cold_flag_persists(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", cold=True))
    with led._conn() as c:
        assert c.execute("SELECT cold FROM requests").fetchone()["cold"] == 1
```

Update the `_row` helper to accept `cold`:

```python
def _row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000, req_type="text", tok_in=10, cold=False):
    return LedgerRow(
        ts_start=100.0, ts_end=101.0, model=model,
        e_window_j=marg_j + 100, e_idle_j=100, e_marginal_j=marg_j,
        kwh_marginal=marg_j / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=cost,
        tok_in=tok_in, tok_out=tok_out, tok_source="backend", energy_confidence="estimated (±15-30%)",
        req_type=req_type, cold=cold,
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ledger.py -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'cold'` (helper), and `model_loads`/`insert_model_load` don't exist.

- [ ] **Step 3: Modify `ledger.py`**

```python
# ledger.py — add cold to the requests schema, and a model_loads table
_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL, ts_end REAL, model TEXT,
    e_window_j REAL, e_idle_j REAL, e_marginal_j REAL,
    kwh_marginal REAL, rate_usd_kwh REAL, cost_marginal_usd REAL,
    tok_in INTEGER, tok_out INTEGER, tok_source TEXT, energy_confidence TEXT,
    req_type TEXT DEFAULT 'text', cold INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS model_loads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, model TEXT, upstream TEXT, load_energy_j REAL, duration_ms REAL, trigger TEXT
);
"""
```

```python
# ledger.py — LedgerRow: add cold as the LAST field with a default
    req_type: str = "text"
    cold: bool = False
```

```python
# ledger.py — _migrate: also add the cold column on old DBs (model_loads is handled by IF NOT EXISTS)
    def _migrate(self, c: sqlite3.Connection) -> None:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(requests)")}
        if "req_type" not in cols:
            c.execute("ALTER TABLE requests ADD COLUMN req_type TEXT DEFAULT 'text'")
        if "cold" not in cols:
            c.execute("ALTER TABLE requests ADD COLUMN cold INTEGER DEFAULT 0")
```

```python
# ledger.py — new methods (add to Ledger)
    def insert_model_load(self, ts: float, model: str, upstream: str,
                          load_energy_j: float, duration_ms: float, trigger: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO model_loads (ts, model, upstream, load_energy_j, duration_ms, trigger) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, model, upstream, load_energy_j, duration_ms, trigger),
            )

    def model_load_summary(self) -> dict:
        sql = "SELECT COUNT(*) AS count, COALESCE(SUM(load_energy_j), 0) AS total_load_j FROM model_loads"
        with self._conn() as c:
            return dict(c.execute(sql).fetchone())
```

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_ledger.py -q && uv run pytest -q`
Expected: PASS. (`insert` uses `asdict`, which now includes `cold`; the schema/migration provide the column.)

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/ledger.py tests/test_ledger.py
git commit -m "feat(ledger): cold flag + model_loads table + summary"
```

---

### Task 3: Proxy — measure TTFT, book the load

**Files:**
- Modify: `src/tokenwatt/proxy.py`
- Modify: `tests/test_proxy.py`
- Modify: `tests/conftest.py` (a slow-first-chunk streaming fixture)

**Interfaces:**
- Consumes: `ColdStartDetector` (Task 1), `Ledger.insert_model_load` + `LedgerRow.cold` (Task 2).
- Produces: `create_app(*, router, meter, idle, ledger, rate, client, detector, _label_factory=..., lifespan=None)` (new keyword-only `detector`). For a streaming request the proxy records TTFT (first-chunk arrival − `t0`); in `_finalize` it calls `detector.observe(...)`; on a cold result it inserts a `model_load` row, subtracts the load energy from the request's marginal (so `e_marginal_j`/kWh/cost are inference-only), and sets `cold=True`. Non-streaming → `ttft=None`, never cold.

- [ ] **Step 1: Add the failing tests + fixture**

In `tests/conftest.py`, add a fixture whose first chunk is delayed (simulating a model load before first token):

```python
import asyncio

@pytest.fixture
def fake_upstream_slow_first_chunk():
    """Streams a content chunk after a delay (simulating a cold model load), then [DONE]."""
    async def chat(request):
        async def gen():
            await asyncio.sleep(0.4)   # "load" delay before the first token
            yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])
```

Append to `tests/test_proxy.py`:

```python
from tokenwatt.coldstart import ColdStartDetector


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
        row = conn.execute("SELECT cold, e_marginal_j FROM requests").fetchone()
    assert row["cold"] == 1                                    # request flagged cold
    assert ml["load_energy_j"] > 100.0                         # a meaningful load chunk was booked
    assert ml["duration_ms"] > 100.0                           # ~0.4s load duration persisted (ms)
    assert 0.0 <= row["e_marginal_j"] < 900.0                  # load subtracted (not the trivial < 1000)


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
```

Also add `from tokenwatt.coldstart import ColdStartDetector` at the TOP of `tests/test_proxy.py`, and update ALL **9** existing `create_app(...)` call sites in that file to pass `detector=ColdStartDetector()` (a fresh detector per test). Verify completeness before running: `grep -c 'detector=' tests/test_proxy.py` must return **11** (9 existing + the 2 new cold tests); if it is < 11, a call site was missed.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_proxy.py -q`
Expected: FAIL — `create_app()` missing required keyword `detector`, then the cold-start assertions.

- [ ] **Step 3: Modify `proxy.py`**

Add the import and the `detector` param:

```python
from tokenwatt.coldstart import ColdStartDetector
```

```python
def create_app(*, router: Router, meter: EnergyMeter, idle: IdleBaseline,
               ledger: Ledger, rate: FlatRate, client: httpx.AsyncClient,
               detector: ColdStartDetector,
               _label_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
               lifespan=None) -> Starlette:
```

In `forward`, add a mutable holder for the first-chunk time (closelse-captured), and rewrite `_finalize` to take the TTFT and apply the detector:

```python
        first_chunk_t: list[float] = []   # records monotonic time of the first streamed chunk

        def _finalize(usage: TokenUsage) -> None:
            idle.request_finished()
            window = meter.end(label)
            dt = time.monotonic() - t0
            idle_e = idle.energy_over(dt)
            marginal_j = (window - idle_e).total_j
            ttft = (first_chunk_t[0] - t0) if first_chunk_t else None
            cold = detector.observe(ledger_model, ttft, marginal_j, dt)
            if cold.is_cold and cold.load_energy_j > 0:
                ledger.insert_model_load(ts=ts_start, model=ledger_model, upstream=route.upstream,
                                         load_energy_j=cold.load_energy_j,
                                         duration_ms=cold.load_time_s * 1000.0, trigger=cold.trigger)
                marginal_j = max(0.0, marginal_j - cold.load_energy_j)
            kwh = marginal_j / 3.6e6
            cost = rate.price(kwh)
            ledger.insert(LedgerRow(
                ts_start=ts_start, ts_end=time.time(), model=ledger_model, req_type=req_type,
                e_window_j=window.total_j, e_idle_j=idle_e.total_j,
                e_marginal_j=marginal_j, kwh_marginal=kwh,
                rate_usd_kwh=rate.usd_per_kwh, cost_marginal_usd=cost,
                tok_in=usage.input if usage else None, tok_out=usage.output if usage else None,
                tok_source=usage.source if usage else "none",
                energy_confidence="estimated (±15-30%)" if usage and usage.source != "none" else "energy-only",
                cold=cold.is_cold,
            ))
```

**Delete `proxy.py` line 16** (`from tokenwatt.cost import marginal_kwh`) as part of this task — its job is inlined here as `kwh = marginal_j / 3.6e6` (so load energy is subtracted before the kWh conversion), and grep confirms it is the only `marginal_kwh` use in `src/`. Do NOT delete the function itself (`cost.py` keeps it for `tests/test_cost.py`).

In the streaming `body_iter`, stamp the first-chunk time:

```python
            async def body_iter():
                try:
                    async for chunk in up_resp.aiter_raw():
                        if not first_chunk_t:
                            first_chunk_t.append(time.monotonic())
                        counter.feed(chunk)
                        yield chunk
                finally:
                    await up_resp.aclose()
                    _finalize(counter.result())
```

The non-streaming branch is unchanged except it never appends to `first_chunk_t`, so `ttft` is `None` there.

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_proxy.py -q && uv run pytest -q`
Expected: PASS (the 2 new cold tests + all existing, now passing `detector=`).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/proxy.py tests/test_proxy.py tests/conftest.py
git commit -m "feat(proxy): measure TTFT, book cold-start model loads, flag cold requests"
```

---

### Task 4: CLI — wire the detector into serve + report the model-load summary

**Files:**
- Modify: `src/tokenwatt/cli.py`
- Modify: `tests/test_report_render.py`

**Interfaces:**
- Consumes: `ColdStartDetector` (Task 1), `Ledger.model_load_summary` (Task 2), modified `create_app` (Task 3).
- Produces: `serve` constructs one `ColdStartDetector()` and passes it to `create_app`. `render_report` adds a model-load summary line.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_report_render.py`:

```python
def test_render_report_shows_model_load_summary(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert_model_load(ts=1000.0, model="m1", upstream="http://a", load_energy_j=480.0, duration_ms=4800.0, trigger="transition")
    text = render_report(led, now=1002.0)
    assert "model loads: 1" in text        # the summary line with the right count
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_report_render.py -q`
Expected: FAIL — no model-load line in the report.

- [ ] **Step 3: Modify `cli.py`**

In `serve`, import and build the detector, and pass it to `create_app`:

```python
    from tokenwatt.coldstart import ColdStartDetector
```

```python
    detector = ColdStartDetector()
```

```python
    app_asgi = create_app(router=Router(cfg.routes), meter=meter, idle=idle, ledger=led,
                          rate=FlatRate(cfg.rate.flat_usd_per_kwh), client=client,
                          detector=detector, lifespan=lifespan)
```

In `render_report`, add a model-load summary line after the 30d line (before the per-model table header):

```python
        f"  last 30d : {month['requests']:>6} req   {month['kwh']:.4f} kWh   {_usd(month['usd'])}",
    ]
    _ml = ledger.model_load_summary()
    if _ml["count"]:
        lines.append(f"  model loads: {_ml['count']} (booked separately: {_ml['total_load_j'] / 3.6e6 * 1000:.3f} Wh)")
    lines += [
        "",
        f"  {'model':<16}{'type':>11}{'req':>6}{'kWh':>12}{'$':>10}{'J/tok':>10}",
    ]
```

(The existing `lines = [...]` list literal is split: keep the header/24h/30d entries in the first list, append the optional model-load line, then `lines += [...]` the blank line + table header. Adjust the surrounding code so the list is built in that order; do not duplicate the table-header line.)

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_report_render.py -q && uv run pytest -q`
Expected: PASS (all). The existing render tests still pass (the model-load line only appears when count > 0).

- [ ] **Step 5: On-device verification (cold-start via LM Studio model swap)**

```bash
cd ~/mlx-cost-project
# point a route at LM Studio (:1234) which JIT-loads/evicts:
#   routes: - {name: lms, type: text, upstream: http://127.0.0.1:1234, match: ["*"]}
uv run tokenwatt serve -c /tmp/tw-lms.yaml --ledger /tmp/tw-cold.sqlite
# in another shell — request model A, then model B (forces an evict+load), streaming:
curl -s http://127.0.0.1:7000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<model-A>","stream":true,"messages":[{"role":"user","content":"hi"}]}' >/dev/null
curl -s http://127.0.0.1:7000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<model-B>","stream":true,"messages":[{"role":"user","content":"hi"}]}' >/dev/null
uv run tokenwatt report --ledger /tmp/tw-cold.sqlite
```

Expected: the swap to model B triggers a cold-start — the report shows a `model loads:` line, and model B's first request is flagged cold with its load energy booked separately. (If LM Studio keeps both resident, force an evict by exceeding its memory budget or setting a short TTL.)

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/cli.py tests/test_report_render.py
git commit -m "feat(cli): wire cold-start detector into serve + model-load summary in report"
```

---

## Self-Review

**1. Spec coverage (M1c-a slice — spec §7 cold-start detection + §10 model_load):**
- Cold-start detection — TTFT-based only; the pre-first-token-*energy* signal from spec §7 is deferred → Task 1. ✓
- Load-energy estimate (time-fraction of window marginal; an estimate, not a measurement) → Task 1, applied in Task 3. ✓
- `model_load` ledger table (with `upstream` + `duration_ms` per spec §10; `ttft_ms` on the *requests* row is deferred to a follow-up) → Task 2. ✓
- `cold` flag on requests → Task 2 (column) + Task 3 (set it). ✓
- Proxy measures TTFT + books the load + keeps the request's per-token cost clean (inference-only marginal) → Task 3. ✓
- Report surfaces model loads → Task 4. ✓
- Backward-compatible migration → Task 2 (`cold` column + `model_loads` via IF NOT EXISTS). ✓
- Streaming-only split, non-streaming unflagged → Task 1 (`ttft None` → not cold), Task 3 (non-stream `ttft=None`). ✓ (documented limitation)
- Deferred correctly (NOT M1c-a): the shareable cost story is M1c-b; Anthropic dialect extraction stays unused.

**2. Placeholder scan:** no TBD/TODO; every code step complete; commands have expected output. ✓

**3. Type consistency:** `ColdStartDetector(...).observe(model, ttft_s, window_marginal_j, duration_s) -> ColdResult(is_cold, load_energy_j, load_time_s, trigger)` used identically in Tasks 1/3. `LedgerRow.cold: bool = False`; `insert_model_load(ts, model, upstream, load_energy_j, duration_ms, trigger)` and `model_load_summary() -> {count, total_load_j}` used identically in Tasks 2/3/4. `create_app(*, ..., detector)` signature matches Task 3 tests + Task 4 serve. The `marginal_kwh` import is deleted from `proxy.py` (Task 3 inlines the kWh conversion to subtract load first). ✓

**Known on-device-only gaps:** the cold-start trigger needs a real load-on-demand backend (LM Studio with eviction). Hermetic tests simulate it with a delayed-first-chunk fixture; on-device validation (Task 4 Step 5) drives an actual LM Studio model swap.
