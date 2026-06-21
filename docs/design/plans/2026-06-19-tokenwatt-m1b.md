# TokenWatt M1b — Request Types (vision + embeddings) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Account for vision and embedding requests correctly — classify each request as text / vision / embedding, record the type in the ledger, and report the right per-token denominator (embeddings are input-only; generative/vision use output tokens) — so a mixed m1/v1/embeddings setup gets honest per-type numbers instead of treating everything as text generation.

**Architecture:** A tiny `classify_request(path, body)` decides type from the request shape (embeddings path → embedding; image content parts → vision; else text). The proxy stamps that type onto the `LedgerRow` (new `req_type` column, added by a forward-only migration so existing DBs keep working). The ledger rollup groups by `(model, req_type)` and picks the denominator per type (`total_in` for embeddings, `total_out` otherwise); the report shows the type column. Token *extraction* is unchanged — the existing backend-usage path already yields input-only usage for embeddings and full usage for vision (mlx-vlm reports it).

**Tech Stack:** Python 3.12, existing stack (no new deps).

## Global Constraints

- **Python ≥ 3.12.** No new dependencies.
- **Request-type values:** exactly `"text"`, `"vision"`, `"embedding"` (singular). Classification: embeddings *path* → `embedding`; any message with an image content part → `vision`; else `text`.
- **`req_type` (ledger: `text`/`vision`/`embedding`, singular) is derived SOLELY by `classify_request(path, body)`** — intentionally NOT taken from `RouteConfig.type` (config namespace is `text`/`vision`/`embeddings`, plural). Do not unify the two namespaces.
- **Per-type denominator (honesty):** J/token uses **output** tokens for `text`/`vision`, **input** tokens for `embedding` (embeddings have no output). When the denominator is 0/None, J/token is `None`, never a divide-by-zero.
- **Backward-compatible ledger:** add the `req_type` column via a forward-only migration (`ALTER TABLE ... ADD COLUMN` when missing) so a pre-M1b `ledger.sqlite` still opens and inserts. New rows default `req_type="text"` when not set.
- **Byte-exact passthrough preserved.** Routing (M1a) and energy measurement (M0) are unchanged; M1b only adds classification + accounting metadata.
- **No sudo at runtime.** Version is read from `VERSION` (auto-bumped per commit). Conventional Commit prefixes.

---

## File Structure

```
src/tokenwatt/reqtype.py    # classify_request(path, body) -> "text"|"vision"|"embedding"
src/tokenwatt/ledger.py     # MODIFIED: LedgerRow gains req_type; schema + migration; by_model groups by (model, req_type) with per-type J/token
src/tokenwatt/proxy.py      # MODIFIED: classify each request, stamp req_type on the LedgerRow
src/tokenwatt/cli.py        # MODIFIED: render_report shows the type column + per-type J/token
tests/test_reqtype.py
tests/test_ledger.py        # MODIFIED: j_per_out_token -> j_per_token; add per-type denominator test
tests/test_proxy.py         # MODIFIED: add vision + embedding req_type recording tests
tests/test_report_render.py # MODIFIED: add a per-type render assertion
```

---

### Task 1: Request-type classifier

**Files:**
- Create: `src/tokenwatt/reqtype.py`
- Modify: `src/tokenwatt/config.py` (one-line stale-comment cleanup, Step 5)
- Test: `tests/test_reqtype.py`

**Interfaces:**
- Produces: `classify_request(path: str, body: dict) -> str` returning `"text"`, `"vision"`, or `"embedding"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reqtype.py
from tokenwatt.reqtype import classify_request


def test_embeddings_path_is_embedding():
    assert classify_request("embeddings", {"input": "hi"}) == "embedding"
    assert classify_request("v1/embeddings", {"input": "hi"}) == "embedding"


def test_image_content_part_is_vision():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    assert classify_request("chat/completions", body) == "vision"


def test_plain_chat_is_text():
    assert classify_request("chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]}) == "text"


def test_list_content_without_image_is_text():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert classify_request("chat/completions", body) == "text"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_reqtype.py -q`
Expected: FAIL — `No module named tokenwatt.reqtype`.

- [ ] **Step 3: Implement `reqtype.py`**

```python
# src/tokenwatt/reqtype.py
from __future__ import annotations

_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def classify_request(path: str, body: dict) -> str:
    """Classify an OpenAI-style request as 'embedding', 'vision', or 'text'."""
    if path.rstrip("/").endswith("embeddings"):
        return "embedding"
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                    return "vision"
    return "text"
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_reqtype.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Fix the now-stale `dialect` comment** (doc cleanup — M1b defers dialect extraction)

In `src/tokenwatt/config.py`, change the `dialect` field comment:
- from: `    dialect: Literal["openai", "anthropic"] = "openai"   # accepted now; used for token extraction in M1b`
- to:   `    dialect: Literal["openai", "anthropic"] = "openai"   # accepted now; dialect-specific token extraction deferred to a later milestone`

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/reqtype.py tests/test_reqtype.py src/tokenwatt/config.py
git commit -m "feat(reqtype): classify requests as text/vision/embedding; fix stale dialect comment"
```

---

### Task 2: Ledger `req_type` column + migration + per-type rollup

**Files:**
- Modify: `src/tokenwatt/ledger.py`
- Modify: `tests/test_ledger.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LedgerRow` gains `req_type: str = "text"` (last field, default). `Ledger.__init__` runs a forward-only migration adding the `req_type` column if missing. `by_model()` now groups by `(model, req_type)` and returns each row with `req_type`, `total_in`, and `j_per_token` (= `total_marginal_j / total_in` for `embedding`, else `/ total_out`; `None` when the denominator is 0). NOTE: the old key `j_per_out_token` is renamed to `j_per_token`.

- [ ] **Step 1: Update + add the failing tests**

In `tests/test_ledger.py`, the `_row` helper and assertions reference the old shape. First, add `req_type` to the `_row` helper signature and pass it through:

```python
def _row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000, req_type="text", tok_in=10):
    return LedgerRow(
        ts_start=100.0, ts_end=101.0, model=model,
        e_window_j=marg_j + 100, e_idle_j=100, e_marginal_j=marg_j,
        kwh_marginal=marg_j / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=cost,
        tok_in=tok_in, tok_out=tok_out, tok_source="backend", energy_confidence="estimated (±15-30%)",
        req_type=req_type,
    )
```

In `test_insert_and_by_model_rollup`, change the J/token assertion key from `j_per_out_token` to `j_per_token`:

```python
    assert abs(r["j_per_token"] - (7_200_000.0 / 2000)) < 1e-6
```

Then append a new test for the embedding denominator:

```python
def test_embedding_j_per_token_uses_input(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    # an embedding row: tok_out is None (no output), tok_in carries the embedded tokens.
    led.insert(_row(model="embed", req_type="embedding", tok_out=None, tok_in=500, marg_j=1000.0))
    r = [x for x in led.by_model() if x["model"] == "embed"][0]
    assert r["req_type"] == "embedding"
    assert abs(r["j_per_token"] - (1000.0 / 500)) < 1e-9   # J / INPUT token for embeddings
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ledger.py -q`
Expected: FAIL — `TypeError: LedgerRow.__init__() got an unexpected keyword argument 'req_type'` (every test using the `_row` helper errors at construction; the field is added in Step 3). Once the field exists, the renamed `j_per_token` rollup assertions are what fail next.

- [ ] **Step 3: Modify `ledger.py`**

Add `req_type TEXT` to the schema, add `req_type` to `LedgerRow` (last field, default), add a migration, and rewrite `by_model`:

```python
# ledger.py — schema: add req_type to the CREATE TABLE column list
_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL, ts_end REAL, model TEXT,
    e_window_j REAL, e_idle_j REAL, e_marginal_j REAL,
    kwh_marginal REAL, rate_usd_kwh REAL, cost_marginal_usd REAL,
    tok_in INTEGER, tok_out INTEGER, tok_source TEXT, energy_confidence TEXT,
    req_type TEXT DEFAULT 'text'
);
"""
```

```python
# ledger.py — LedgerRow: add req_type as the LAST field with a default
@dataclass
class LedgerRow:
    ts_start: float
    ts_end: float
    model: str
    e_window_j: float
    e_idle_j: float
    e_marginal_j: float
    kwh_marginal: float
    rate_usd_kwh: float | None
    cost_marginal_usd: float | None
    tok_in: int | None
    tok_out: int | None
    tok_source: str
    energy_confidence: str
    req_type: str = "text"
```

```python
# ledger.py — Ledger.__init__: run the migration after creating the table
    def __init__(self, path: str) -> None:
        self._path = path
        with self._conn() as c:
            c.executescript(_SCHEMA)
            self._migrate(c)

    def _migrate(self, c: sqlite3.Connection) -> None:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(requests)")}
        if "req_type" not in cols:        # forward-only: pre-M1b DBs gain the column
            c.execute("ALTER TABLE requests ADD COLUMN req_type TEXT DEFAULT 'text'")
```

```python
# ledger.py — by_model: group by (model, req_type), per-type denominator
    def by_model(self) -> list[dict]:
        sql = """
        SELECT model, req_type,
               COUNT(*)                            AS requests,
               COALESCE(SUM(kwh_marginal), 0)      AS total_kwh,
               SUM(cost_marginal_usd)              AS total_usd,
               COALESCE(SUM(tok_out), 0)           AS total_out,
               COALESCE(SUM(tok_in), 0)            AS total_in,
               COALESCE(SUM(e_marginal_j), 0)      AS total_marginal_j
        FROM requests GROUP BY model, req_type ORDER BY total_usd DESC
        """
        out = []
        with self._conn() as c:
            for r in c.execute(sql):
                d = dict(r)
                denom = d["total_in"] if d["req_type"] == "embedding" else d["total_out"]
                d["j_per_token"] = (d["total_marginal_j"] / denom) if denom else None
                out.append(d)
        return out
```

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_ledger.py -q && uv run pytest -q`
Expected: PASS. (Existing rows default to `req_type="text"`, so the prior rollup assertions still hold under the renamed key.)

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/ledger.py tests/test_ledger.py
git commit -m "feat(ledger): req_type column + migration + per-type J/token denominator"
```

---

### Task 3: Proxy stamps the request type

**Files:**
- Modify: `src/tokenwatt/proxy.py`
- Modify: `tests/test_proxy.py`

**Interfaces:**
- Consumes: `classify_request` (Task 1), `LedgerRow.req_type` (Task 2).
- Produces: every ledger row the proxy writes carries `req_type = classify_request(path, req_json)`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_proxy.py` (the imports already include `RouteConfig`, `Router`, `FakeMeter`, `IdleBaseline`, `FlatRate`, `Ledger`, `_client_for`):

```python
async def test_vision_request_records_req_type(tmp_path, fake_upstream_json):
    router = Router([RouteConfig(name="v1", type="vision", upstream="http://up", match=["*"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json))
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
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/embeddings", json={"model": "embed", "input": "hello world"})
    with ledger._conn() as conn:
        assert conn.execute("SELECT req_type FROM requests").fetchone()["req_type"] == "embedding"
```

(Note: the `fake_upstream_json` fixture only routes `/v1/chat/completions`. Add an `/v1/embeddings` route to it OR use a permissive fixture. See Step 3's conftest note.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_proxy.py -k "req_type" -q`
Expected: FAIL — `no such column: req_type` is gone (Task 2 added it), so failure is the assertion (`text` recorded, not `vision`/`embedding`) — the proxy doesn't classify yet.

- [ ] **Step 3: Modify `proxy.py` and the conftest fixture**

In `tests/conftest.py`, make `fake_upstream_json` also answer `/v1/embeddings` (so the embeddings test routes somewhere). Change its route list to:

```python
    return Starlette(routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", chat, methods=["POST"]),
    ])
```

In `proxy.py`, import the classifier and stamp the type. Add the import:

```python
from tokenwatt.reqtype import classify_request
```

In `forward`, right after `is_stream = bool(...)`, compute the type (it's before the route-resolution 404; classification doesn't need a route):

```python
        req_type = classify_request(path, req_json)
```

In the `_finalize` closure's `LedgerRow(...)`, add the field:

```python
                req_type=req_type,
```

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_proxy.py -q && uv run pytest -q`
Expected: PASS (the 2 new req_type tests plus all existing; the existing text requests record `req_type="text"`).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/proxy.py tests/test_proxy.py tests/conftest.py
git commit -m "feat(proxy): classify and record request type (text/vision/embedding)"
```

---

### Task 4: Report shows type + per-type J/token

**Files:**
- Modify: `src/tokenwatt/cli.py`
- Modify: `tests/test_report_render.py`

**Interfaces:**
- Consumes: `by_model()` rows with `req_type` + `j_per_token` (Task 2).
- Produces: `render_report` prints a `type` column and the per-type `J/tok`.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_report_render.py` (imports already include `Ledger`, `LedgerRow`, `render_report`):

```python
def test_render_report_shows_type_and_embedding_j_per_token(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="embed",
        e_window_j=1001.0, e_idle_j=1.0, e_marginal_j=1000.0,
        kwh_marginal=1000.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0001,
        tok_in=500, tok_out=None, tok_source="backend", energy_confidence="estimated (±15-30%)",
        req_type="embedding",
    ))
    text = render_report(led, now=1002.0)
    embed_line = next(l for l in text.splitlines() if "embed" in l)
    assert "embedding" in embed_line    # the type column is shown for this row
    assert "2.000" in embed_line        # J/tok = 1000 / 500 INPUT tokens; renders '-' if denom wrongly used tok_out=None
    assert "J/tok" in text              # column header present
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_report_render.py -q`
Expected: FAIL — the renderer references the old `j_per_out_token` key (KeyError) and shows no type column.

- [ ] **Step 3: Modify `render_report` in `cli.py`**

Replace the per-model header line and loop:

```python
        f"  {'model':<16}{'type':>11}{'req':>6}{'kWh':>12}{'$':>10}{'J/tok':>10}",
    ]
    for r in ledger.by_model():
        jpt = f"{r['j_per_token']:.3f}" if r["j_per_token"] is not None else "-"
        lines.append(
            f"  {r['model']:<16}{r['req_type']:>11}{r['requests']:>6}{r['total_kwh']:>12.4f}"
            f"{_usd(r['total_usd']):>10}{jpt:>10}"
        )
    return "\n".join(lines)
```

(The two pre-existing render tests assert on `"m1"`, `"$0.31"`, `"estimated"`, `"—"`, and `"$0.0000" not in text` — all still hold with the new columns.)

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_report_render.py -q && uv run pytest -q`
Expected: PASS (all).

- [ ] **Step 5: On-device verification (vision routing + per-type accounting)**

```bash
cd ~/mlx-cost-project
# config routing m1 (text) + v1 (vision) to the LAN backends (edit IP/ports):
uv run tokenwatt serve -c /tmp/tw-m1a.yaml --ledger /tmp/tw-m1b.sqlite
# in another shell — a VISION request (image) to v1, and a TEXT request to m1:
curl -s http://127.0.0.1:7000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"mlx-community/Qwen3-VL-8B-Instruct-8bit","max_tokens":20,
  "messages":[{"role":"user","content":[
    {"type":"text","text":"What colour is this?"},
    {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Solid_red.svg/64px-Solid_red.svg.png"}}]}]}' >/dev/null
curl -s http://127.0.0.1:7000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.6-27b","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}' >/dev/null
uv run tokenwatt report --ledger /tmp/tw-m1b.sqlite
```

Expected: the report shows the `v1` row typed **vision** and the `m1` row typed **text**, each with its own J/tok. (Embeddings need a running embeddings backend; covered hermetically here, validate on-device when one is up.)

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/cli.py tests/test_report_render.py
git commit -m "feat(report): per-type rollup column + per-type J/token"
```

---

## Self-Review

**1. Spec coverage (M1b slice — spec §6 request-type classifier, §10 ledger, §16 M1 row "embeddings + vision request-types … per-type rollups"):**
- Request-type classifier (image part → vision; embeddings path → embedding; else text) → Task 1. ✓
- Ledger records request type → Task 2 (`req_type` column + migration). ✓
- Per-type denominator (embeddings = input tokens; else output) → Task 2 (`by_model` `j_per_token`). ✓
- Proxy stamps the type → Task 3. ✓
- Per-type rollups in the report → Task 4. ✓
- Backward-compatible ledger (existing DBs) → Task 2 (`_migrate`). ✓
- Token extraction unchanged (backend usage already yields input-only for embeddings, full for vision) — relied on, not re-implemented. Vision J/token relies on backend-reported usage (mlx-vlm provides it); `SelfCounter` only counts string content, so a list-content vision message self-counts input=0 — if backend usage is ever absent, the vision row honestly falls to energy-only / `j_per_token=None` rather than a bogus self-count. ✓
- Deferred correctly (NOT M1b): cold-start/model_load booking, README GIF, `$/Mtok` comparison card (M1c); `dialect`-based Anthropic token extraction (the `dialect` field stays accepted-but-unused).

**2. Placeholder scan:** no TBD/TODO; every code step complete; commands have expected output. ✓

**3. Type consistency:** `classify_request(path, body) -> str` used identically in Task 1 + Task 3. `LedgerRow` gains `req_type: str = "text"` (default keeps all existing constructions valid). `by_model()` returns `req_type` + `j_per_token` (renamed from `j_per_out_token`) — every consumer updated: Task 2 tests, Task 4 `render_report`. No remaining reference to `j_per_out_token`. ✓

**Known on-device-only gaps:** the vision on-device test needs v1 up (it is); embeddings on-device needs an embeddings backend (not currently running) — embeddings are covered hermetically, validate on-device when one is available.
