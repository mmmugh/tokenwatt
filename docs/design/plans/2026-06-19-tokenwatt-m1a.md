# TokenWatt M1a — Config + Multi-Backend Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let TokenWatt front several local backends at once through one port — a YAML config with fail-loud validation, a `model → upstream` router with explicit precedence, `tokenwatt init` scaffolding, all wired into the proxy so a request is routed by its `model` field and the ledger groups by the route's canonical name.

**Architecture:** A Pydantic `Config` (parsed from YAML) holds a list of `RouteConfig`s. A `Router` resolves a request's `model` string to a route by precedence (exact > longest-prefix glob > `*` catch-all, first-in-list breaks ties). The proxy's `create_app` takes a `Router` instead of a single `upstream`; it resolves the route, forwards the original bytes to that route's upstream, and records the route's `name` (not the raw model id) in the ledger. Unmatched models get a fail-loud 404; the zero-config default ships a `*` catch-all so the default experience never 404s.

**Tech Stack:** Python 3.12, `pydantic>=2`, `pyyaml`, `fnmatch` (stdlib), plus the existing Starlette/httpx/typer/zeus stack.

## Global Constraints

- **Python ≥ 3.12.** New deps: `pydantic>=2`, `pyyaml>=6`.
- **Config is YAML** (`tokenwatt.yaml`); the bulk is the `routes:` list-of-records.
- **Zero-config boots:** `serve` with no `--config` and no `--upstream` uses `default_config()` — a single `*` catch-all route to `http://127.0.0.1:8080`, port 7000, host 127.0.0.1, ledger `~/.tokenwatt/ledger.sqlite`, no rate (cost labeled estimated).
- **Validation is fail-loud (Pydantic):** a bad config dies before the proxy binds, naming the field path + message. Duplicate route names and non-URL upstreams halt startup. Never silently forward a misrouted request.
- **Routing precedence (exact values):** `exact > longest-prefix/glob > "*" catch-all`; ties broken by first-in-list (earlier route, then earlier pattern). The route `name` is the canonical key the ledger/report group by.
- **No-match is a 404** (`{"error":{"type":"no_route"}}`), not a silent forward — a misrouted backend poisons the ledger. (The energy-only fallback for unknown *response dialect* on a matched route is unchanged from M0 and out of M1a scope.)
- **Byte-exact passthrough is preserved:** the original request bytes (`content=raw`, with the client's original `model` id) are forwarded unchanged; only the ledger records the route name.
- **No sudo at runtime.** Version is read from `VERSION` (auto-bumped per commit). Conventional Commit prefixes.

---

## File Structure

```
src/tokenwatt/config.py     # RouteConfig/RateConfig/Config (Pydantic) + load_config/default_config/ConfigError
src/tokenwatt/router.py     # Router.resolve(model) -> RouteConfig | None  (precedence)
src/tokenwatt/proxy.py      # MODIFIED: create_app takes router= instead of upstream=
src/tokenwatt/cli.py        # MODIFIED: serve loads config/builds router; new `init` command
src/tokenwatt/data/m1-v1-embeddings.yaml   # in-package example config (ships in wheel; loaded via importlib.resources)
tests/test_config.py        # model validation
tests/test_config_loader.py # YAML load + fail-loud formatting + defaults
tests/test_router.py        # precedence matching
tests/test_proxy.py         # MODIFIED: existing tests use router=; + no-route 404 + route-name recording
tests/test_cli_config.py    # `init` scaffolds a config that round-trips through load_config
```

---

### Task 1: Config models (Pydantic) + validation

**Files:**
- Create: `src/tokenwatt/config.py`
- Modify: `pyproject.toml` (add `pydantic>=2`, `pyyaml>=6`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `RouteConfig` (Pydantic `BaseModel`): `name: str`, `type: Literal["text","vision","embeddings"] = "text"`, `upstream: str` (validated to start with `http://`/`https://`, trailing slash stripped), `dialect: Literal["openai","anthropic"] = "openai"` (accepted now, used for token extraction in M1b), `match: list[str]` (≥1 item).
  - `RateConfig`: `flat_usd_per_kwh: float | None = None`.
  - `Config`: `port: int = 7000`, `host: str = "127.0.0.1"`, `ledger: str = "~/.tokenwatt/ledger.sqlite"`, `rate: RateConfig` (default factory), `routes: list[RouteConfig]` (default factory); rejects duplicate route names.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from pydantic import ValidationError
from tokenwatt.config import Config, RouteConfig


def test_valid_config_parses_with_defaults():
    c = Config(routes=[{"name": "m1", "upstream": "http://127.0.0.1:8080/", "match": ["m1"]}])
    assert c.port == 7000 and c.host == "127.0.0.1"
    assert c.routes[0].type == "text"
    assert c.routes[0].upstream == "http://127.0.0.1:8080"   # trailing slash stripped


def test_bad_upstream_rejected():
    with pytest.raises(ValidationError) as e:
        RouteConfig(name="m1", upstream="127.0.0.1:8080", match=["m1"])
    assert "http://" in str(e.value)


def test_bad_type_rejected():
    with pytest.raises(ValidationError) as e:
        RouteConfig(name="m1", type="embedding", upstream="http://x", match=["m1"])  # missing 's'
    assert "embeddings" in str(e.value)   # message names the allowed values (helpful, fail-loud)


def test_empty_match_rejected():
    with pytest.raises(ValidationError):
        RouteConfig(name="m1", upstream="http://x", match=[])


def test_duplicate_route_names_rejected():
    with pytest.raises(ValidationError) as e:
        Config(routes=[
            {"name": "m1", "upstream": "http://a", "match": ["a"]},
            {"name": "m1", "upstream": "http://b", "match": ["b"]},
        ])
    assert "duplicate" in str(e.value).lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `No module named tokenwatt.config`.

- [ ] **Step 3: Add deps and implement `config.py` (models only)**

In `pyproject.toml`, add to `dependencies`:

```toml
    "pydantic>=2",
    "pyyaml>=6",
```

```python
# src/tokenwatt/config.py
from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RouteConfig(BaseModel):
    name: str
    type: Literal["text", "vision", "embeddings"] = "text"
    dialect: Literal["openai", "anthropic"] = "openai"   # accepted now; used for token extraction in M1b
    upstream: str
    match: list[str]

    @field_validator("upstream")
    @classmethod
    def _check_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"upstream must start with http:// or https:// (got {v!r})")
        return v.rstrip("/")

    @field_validator("match")
    @classmethod
    def _check_match(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("match must list at least one pattern")
        return v


class RateConfig(BaseModel):
    flat_usd_per_kwh: float | None = None


class Config(BaseModel):
    port: int = 7000
    host: str = "127.0.0.1"
    ledger: str = "~/.tokenwatt/ledger.sqlite"
    rate: RateConfig = Field(default_factory=RateConfig)
    routes: list[RouteConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> "Config":
        names = [r.name for r in self.routes]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate route name(s): {dupes}")
        return self
```

- [ ] **Step 4: Install deps and run the tests**

Run: `uv pip install -e ".[dev]" && uv run pytest tests/test_config.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/tokenwatt/config.py tests/test_config.py
git commit -m "feat(config): pydantic Config/RouteConfig models with validation"
```

---

### Task 2: YAML loader + fail-loud errors + zero-config default

**Files:**
- Modify: `src/tokenwatt/config.py`
- Test: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: `Config`, `RouteConfig` (Task 1).
- Produces:
  - `ConfigError(Exception)`.
  - `default_config() -> Config` — one `*` catch-all `RouteConfig(name="default", type="text", upstream="http://127.0.0.1:8080", match=["*"])`.
  - `load_config(path: str | None) -> Config` — `None` → `default_config()`; otherwise expand `~`, read YAML, validate; raises `ConfigError` (with field path) on a missing file, malformed YAML, or validation failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_loader.py
import pytest
from tokenwatt.config import load_config, default_config, ConfigError


def test_default_config_has_catchall():
    c = default_config()
    assert c.routes[0].match == ["*"]
    assert c.routes[0].upstream == "http://127.0.0.1:8080"


def test_load_none_returns_default():
    assert load_config(None).routes[0].match == ["*"]


def test_load_valid_yaml(tmp_path):
    p = tmp_path / "tw.yaml"
    p.write_text(
        "port: 9000\n"
        "routes:\n"
        "  - name: m1\n"
        "    upstream: http://127.0.0.1:8080\n"
        "    match: [m1, 'mlx-community/Qwen3-*']\n"
    )
    c = load_config(str(p))
    assert c.port == 9000 and c.routes[0].name == "m1"
    assert c.routes[0].match == ["m1", "mlx-community/Qwen3-*"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_config(str(tmp_path / "nope.yaml"))
    assert "not found" in str(e.value)


def test_invalid_route_reports_field_path(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("routes:\n  - name: m1\n    upstream: not-a-url\n    match: [m1]\n")
    with pytest.raises(ConfigError) as e:
        load_config(str(p))
    msg = str(e.value)
    assert "routes" in msg and "upstream" in msg   # names the offending field path
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config_loader.py -q`
Expected: FAIL — `cannot import name 'load_config'`.

- [ ] **Step 3: Add the loader to `config.py`**

```python
# append to src/tokenwatt/config.py
import os

import yaml
from pydantic import ValidationError


class ConfigError(Exception):
    pass


def default_config() -> Config:
    return Config(routes=[
        RouteConfig(name="default", type="text",
                    upstream="http://127.0.0.1:8080", match=["*"])
    ])


def _format_validation_error(path: str, err: ValidationError) -> str:
    lines = [f"invalid config in {path}:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def load_config(path: str | None) -> Config:
    if path is None:
        return default_config()
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(expanded) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}")
    try:
        return Config(**data)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(path, e))
```

Put `import os`, `import yaml`, and `from pydantic import ValidationError` at the TOP of `config.py` with the Task 1 imports (the block above shows them inline only for clarity). Add `ValidationError` once — do not duplicate the existing `from pydantic import BaseModel, Field, field_validator, model_validator` line.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config_loader.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/config.py tests/test_config_loader.py
git commit -m "feat(config): YAML loader, zero-config default, fail-loud errors"
```

---

### Task 3: Router (precedence matching)

**Files:**
- Create: `src/tokenwatt/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `RouteConfig` (Task 1).
- Produces: `Router(routes: list[RouteConfig])` with `resolve(model: str) -> RouteConfig | None`. Precedence: exact match (kind 2) > glob with longest literal prefix (kind 1) > `*` catch-all (kind 0); ties → earliest route, then earliest pattern. Returns `None` if nothing matches.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
from tokenwatt.config import RouteConfig
from tokenwatt.router import Router


def _r(name, *match):
    return RouteConfig(name=name, upstream="http://x", match=list(match))


def test_exact_beats_glob_and_catchall():
    router = Router([_r("cap", "*"), _r("glob", "mlx-*"), _r("exact", "mlx-7b")])
    assert router.resolve("mlx-7b").name == "exact"


def test_glob_beats_catchall_and_longest_prefix_wins():
    router = Router([_r("cap", "*"), _r("short", "mlx-*"), _r("long", "mlx-community/*")])
    assert router.resolve("mlx-community/Qwen").name == "long"   # longer literal prefix
    assert router.resolve("mlx-7b").name == "short"


def test_first_in_list_breaks_ties():
    router = Router([_r("a", "mlx-*"), _r("b", "mlx-*")])
    assert router.resolve("mlx-7b").name == "a"


def test_catchall_matches_anything_and_none_when_unmatched():
    assert Router([_r("cap", "*")]).resolve("whatever").name == "cap"
    assert Router([_r("only", "m1")]).resolve("m2") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_router.py -q`
Expected: FAIL — `No module named tokenwatt.router`.

- [ ] **Step 3: Implement `router.py`**

```python
# src/tokenwatt/router.py
from __future__ import annotations
import fnmatch

from tokenwatt.config import RouteConfig


class Router:
    def __init__(self, routes: list[RouteConfig]) -> None:
        self._routes = routes

    def resolve(self, model: str) -> RouteConfig | None:
        best_route: RouteConfig | None = None
        best_key = None
        for ri, route in enumerate(self._routes):
            for pi, pattern in enumerate(route.match):
                score = self._score(pattern, model)
                if score is None:
                    continue
                # earlier route / earlier pattern win ties -> negate indices so larger wins
                key = (score[0], score[1], -ri, -pi)
                if best_key is None or key > best_key:
                    best_key, best_route = key, route
        return best_route

    @staticmethod
    def _score(pattern: str, model: str):
        if pattern == "*":
            return (0, 0)                       # catch-all
        if "*" in pattern:
            if fnmatch.fnmatchcase(model, pattern):
                return (1, len(pattern.split("*", 1)[0]))   # glob; longer prefix = more specific
            return None
        if pattern == model:
            return (2, len(pattern))            # exact
        return None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_router.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/router.py tests/test_router.py
git commit -m "feat(router): model->route resolution with exact/glob/catchall precedence"
```

---

### Task 4: Proxy integration (router replaces single upstream)

**Files:**
- Modify: `src/tokenwatt/proxy.py`
- Modify: `tests/test_proxy.py` (existing tests switch to `router=`; add 2 new)

**Interfaces:**
- Consumes: `Router` (Task 3), `RouteConfig` (Task 1), everything proxy already used.
- Produces: `create_app(*, router: Router, meter, idle, ledger, rate, client, _label_factory=..., lifespan=None) -> Starlette`. Resolves the request `model` to a route; unmatched → HTTP 404 `{"error":{"type":"no_route"}}` (no energy window opened); matched → forward original bytes to `route.upstream`, record `route.name` as the ledger `model`.

- [ ] **Step 1: Update the existing proxy tests + add new ones (write them first; they will fail to compile against the old signature)**

At the top of `tests/test_proxy.py`, add the import and a shared router, and replace every `upstream="http://up"` argument with `router=_ROUTER`:

```python
# add to imports in tests/test_proxy.py
from tokenwatt.config import RouteConfig
from tokenwatt.router import Router

_ROUTER = Router([RouteConfig(name="m1", type="text", upstream="http://up", match=["*"])])
```

In each existing `create_app(...)` call, replace `upstream="http://up",` with `router=_ROUTER,`. (There are 5 call sites: streaming-byte-exact, json-backend-usage, upstream-error, cost-none, streaming-prefers-backend-usage.)

Then append two new tests:

```python
async def test_unmatched_model_returns_404(tmp_path, fake_upstream_json):
    router = Router([RouteConfig(name="m1", type="text", upstream="http://up", match=["only-m1"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    meter = FakeMeter()
    app = create_app(router=router, meter=meter, idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "something-else", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert meter._open == set()   # a 404 must NOT open/leak an energy window
    with ledger._conn() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"] == 0


async def test_ledger_records_route_name_not_raw_model(tmp_path, fake_upstream_json):
    # client sends a long HF id; the route named "v1" matches it by glob -> ledger groups by "v1".
    router = Router([RouteConfig(name="v1", type="vision", upstream="http://up", match=["*VL*"])])
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                     ledger=ledger, rate=FlatRate(0.31), client=_client_for(fake_upstream_json))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "mlx-community/Qwen3-VL-8B", "messages": [{"role": "user", "content": "hi"}]})
    assert ledger.by_model()[0]["model"] == "v1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_proxy.py -q`
Expected: FAIL — `create_app()` got an unexpected keyword `router` (or missing `upstream`), and the 2 new tests error.

- [ ] **Step 3: Modify `create_app` in `proxy.py`**

Change the import block to add the Router type, change the signature, and resolve the route. Replace the signature line and the `model =` / forward / `_finalize` lines:

```python
# proxy.py — change the import near the top:
from tokenwatt.router import Router
```

```python
# proxy.py — new signature (was: def create_app(*, upstream: str, ...)):
def create_app(*, router: Router, meter: EnergyMeter, idle: IdleBaseline,
               ledger: Ledger, rate: FlatRate, client: httpx.AsyncClient,
               _label_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
               lifespan=None) -> Starlette:
```

Inside `forward`, **immediately after `is_stream = bool(req_json.get("stream"))` (current proxy.py line 37) and BEFORE `label = _label_factory()` / `meter.begin(label)`** — so an unmatched request never opens an energy window — resolve the route and bail out:

```python
        route = router.resolve(model)
        if route is None:
            return JSONResponse(
                {"error": {"message": f"no route for model {model!r}", "type": "no_route"}},
                status_code=404,
            )
        ledger_model = route.name
```

Change the upstream URL in `build_request` from `f"{upstream}/v1/{path}"` to `f"{route.upstream}/v1/{path}"`, and in `_finalize`'s `LedgerRow(...)` change `model=model` to `model=ledger_model`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_proxy.py -q`
Expected: PASS (all 7 — 5 updated + 2 new).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (everything; the only consumer of `create_app` besides tests is `cli.py serve`, updated in Task 5 — until then `serve` is broken but untested, which Task 5 fixes).

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/proxy.py tests/test_proxy.py
git commit -m "feat(proxy): route requests via Router; record route name; 404 on no match"
```

---

### Task 5: CLI — config-driven `serve` + `init` + example config

**Files:**
- Modify: `src/tokenwatt/cli.py`
- Modify: `pyproject.toml` (add `[tool.setuptools.package-data]`)
- Create: `src/tokenwatt/data/m1-v1-embeddings.yaml`
- Test: `tests/test_cli_config.py`

**Interfaces:**
- Consumes: `load_config`, `RouteConfig`, `ConfigError` (Tasks 1–2), `Router` (Task 3), modified `create_app` (Task 4).
- Produces: `serve` now takes `--config PATH` (loads it; `--upstream/--port/--host/--rate/--ledger` override); `init` writes the example config; `EXAMPLE_CONFIG: str` constant.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_config.py
from typer.testing import CliRunner
from tokenwatt.cli import app
from tokenwatt.config import load_config

runner = CliRunner()


def test_init_writes_config_that_round_trips(tmp_path):
    cfg = tmp_path / "tw.yaml"
    res = runner.invoke(app, ["init", "--config", str(cfg)])
    assert res.exit_code == 0 and cfg.exists()
    # the scaffolded file must parse cleanly and contain the three example routes
    c = load_config(str(cfg))
    names = {r.name for r in c.routes}
    assert {"m1", "v1", "embeddings"} <= names


def test_init_refuses_overwrite_without_force(tmp_path):
    cfg = tmp_path / "tw.yaml"
    cfg.write_text("port: 1\n")
    res = runner.invoke(app, ["init", "--config", str(cfg)])
    assert res.exit_code != 0
    assert cfg.read_text() == "port: 1\n"   # untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_config.py -q`
Expected: FAIL — `init` command does not exist.

- [ ] **Step 3: Add the example config and rewrite `serve` + add `init` in `cli.py`**

Create the example as **in-package data** at `src/tokenwatt/data/m1-v1-embeddings.yaml` (so it ships in the wheel), and declare it in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
tokenwatt = ["data/*.yaml"]
```

`src/tokenwatt/data/m1-v1-embeddings.yaml`:

```yaml
# tokenwatt.yaml — run:  tokenwatt serve -c tw.yaml
# Every key is optional. With NO config, `tokenwatt serve` still boots and measures
# (port 7000, ~/.tokenwatt/ledger.sqlite, estimated rate, catch-all to :8080).

port: 7000
host: 127.0.0.1
ledger: ~/.tokenwatt/ledger.sqlite

rate:
  flat_usd_per_kwh: 0.31      # omit -> costs labeled 'estimated'

# model -> upstream routes. Match precedence: exact > longest-prefix/glob > '*';
# first-in-list breaks ties. The route `name` is how the ledger/report groups.
routes:
  - name: m1
    type: text
    upstream: http://127.0.0.1:8080
    match: [m1, "mlx-community/Qwen3-Coder*", "mlx-community/Qwen3-*"]
  - name: v1
    type: vision
    dialect: openai            # mlx-vlm speaks openai + anthropic; used for token extraction in M1b
    upstream: http://127.0.0.1:8081
    match: [v1, "mlx-community/Qwen3-VL-*", "*VL*"]
  - name: embeddings
    type: embeddings
    upstream: http://127.0.0.1:8080
    match: [embeddings, "*-embed*", "text-embedding-*"]
```

In `cli.py`, replace the existing `serve` function and add `init` + the `EXAMPLE_CONFIG` constant (read from the shipped example so they can't drift):

```python
# cli.py — add near the other imports
from importlib.resources import files

# Loaded from in-package data so it resolves for BOTH editable and wheel installs
# (a repo-root examples/ dir is NOT in the wheel and would FileNotFoundError at import).
EXAMPLE_CONFIG = files("tokenwatt").joinpath("data/m1-v1-embeddings.yaml").read_text(encoding="utf-8")
```

```python
# cli.py — REPLACE the whole serve() function with:
@app.command()
def serve(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="path to tokenwatt.yaml"),
    upstream: Optional[str] = typer.Option(None, "--upstream", help="single-backend shortcut (overrides config routes)"),
    port: Optional[int] = typer.Option(None, "--port"),
    host: Optional[str] = typer.Option(None, "--host"),
    rate: Optional[float] = typer.Option(None, "--rate", help="flat $/kWh; omit to label costs 'estimated'"),
    ledger: Optional[str] = typer.Option(None, "--ledger"),
):
    """Run the measuring proxy (no sudo)."""
    import os
    from contextlib import asynccontextmanager
    import uvicorn
    from tokenwatt.config import load_config, RouteConfig, ConfigError
    from tokenwatt.router import Router
    from tokenwatt.meter import ZeusMeter
    from tokenwatt.proxy import create_app

    try:
        cfg = load_config(config)
    except ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if upstream is not None:                       # single-backend shortcut
        cfg.routes = [RouteConfig(name="default", type="text", upstream=upstream, match=["*"])]
    if rate is not None:
        cfg.rate.flat_usd_per_kwh = rate
    eff_host = host if host is not None else cfg.host
    eff_port = port if port is not None else cfg.port
    eff_ledger = os.path.expanduser(ledger if ledger is not None else cfg.ledger)

    os.makedirs(os.path.dirname(eff_ledger), exist_ok=True)
    led = Ledger(eff_ledger)
    meter = ZeusMeter()
    idle = IdleBaseline(meter)
    client = httpx.AsyncClient(timeout=None)

    @asynccontextmanager
    async def lifespan(app):
        async def _idle_loop():
            while True:
                idle.sample()
                await asyncio.sleep(2.0)
        task = asyncio.create_task(_idle_loop())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()

    app_asgi = create_app(router=Router(cfg.routes), meter=meter, idle=idle, ledger=led,
                          rate=FlatRate(cfg.rate.flat_usd_per_kwh), client=client, lifespan=lifespan)

    routes_desc = ", ".join(f"{r.name}->{r.upstream}" for r in cfg.routes) or "(none)"
    typer.echo(f"TokenWatt proxy on http://{eff_host}:{eff_port}  routes: {routes_desc}  (no sudo)")
    uvicorn.run(app_asgi, host=eff_host, port=eff_port, log_level="warning")


@app.command()
def init(
    config: str = typer.Option("tokenwatt.yaml", "--config", "-c"),
    force: bool = typer.Option(False, "--force"),
):
    """Scaffold a commented tokenwatt.yaml you can edit."""
    import os
    if os.path.exists(config) and not force:
        typer.echo(f"{config} already exists; pass --force to overwrite", err=True)
        raise typer.Exit(1)
    with open(config, "w") as f:
        f.write(EXAMPLE_CONFIG)
    typer.echo(f"wrote {config}")
```

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_cli_config.py -q && uv run pytest -q`
Expected: PASS (all).

- [ ] **Step 4b: Wheel-install smoke (guards the packaging fix)**

The CliRunner test passes under an editable install even if the data file is not packaged — prove the wheel actually ships `data/*.yaml`:

```bash
uv build
python -m venv /tmp/tw-wheel && /tmp/tw-wheel/bin/pip -q install dist/tokenwatt-*.whl
cd /tmp && /tmp/tw-wheel/bin/tokenwatt init -c /tmp/wheelcheck.yaml && head -1 /tmp/wheelcheck.yaml
```
Expected: `wrote /tmp/wheelcheck.yaml` with NO `FileNotFoundError` (example resolved from in-package data). `cd` back to the repo afterward.

- [ ] **Step 5: On-device verification (multi-backend) — the real acceptance test**

```bash
cd ~/mlx-cost-project
uv run tokenwatt init -c /tmp/tw.yaml            # then edit upstreams to your LAN IP / ports
uv run tokenwatt serve -c /tmp/tw.yaml --ledger /tmp/tw-m1a.sqlite
# in another shell, send to m1 (text) and v1 (vision) by their model ids:
curl -s http://127.0.0.1:7000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}' >/dev/null
uv run tokenwatt report --ledger /tmp/tw-m1a.sqlite   # row grouped under route name "m1"
```

Expected: requests route to the correct upstream by model; the report groups by route `name`; an unknown model returns 404.

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/cli.py pyproject.toml src/tokenwatt/data/m1-v1-embeddings.yaml tests/test_cli_config.py
git commit -m "feat(cli): config-driven serve + init scaffold + in-package example config"
```

---

## Self-Review

**1. Spec coverage (M1a slice of spec §12 + §16 M1 row):**
- YAML config → Tasks 1–2. ✓
- Pydantic fail-loud validation (field path + pydantic message; for the `type` enum the message names the allowed values) → Task 1 validators + Task 2 `_format_validation_error`. ✓ (The spec's optional "did you mean 'embeddings'?" suggestion is deferred polish, not implemented in M1a.)
- Zero-config boot (catch-all to :8080) → Task 2 `default_config`, Task 5 `serve`. ✓
- `tokenwatt init` scaffold (non-destructive) → Task 5. ✓
- Dual-match routing (alias + full-id + glob) with precedence exact>glob>catch-all, first-in-list ties → Task 3. ✓
- Route `name` as canonical ledger key → Task 4 (`ledger_model = route.name`). ✓
- `examples/m1-v1-embeddings.yaml` → Task 5. ✓
- Duplicate-name / bad-URL halt startup → Task 1 validators (surfaced via `load_config`). ✓
- No-match 404 (not silent forward) → Task 4. ✓
- Byte-exact passthrough preserved (original `model` id forwarded; route name only in ledger) → Task 4 (`content=raw` unchanged). ✓
- Deferred correctly (NOT M1a): vision/embeddings request-type accounting (M1b), cold-start/model_load (M1c), README GIF (M1c), `env`-var interpolation, regex DSL.

**2. Placeholder scan:** no TBD/TODO; every code step complete; commands have expected output. ✓

**3. Type consistency:** `RouteConfig(name, type, upstream, match)` constructed identically in Tasks 1/3/4/5 and tests. `Config(port, host, ledger, rate, routes)` consistent. `Router(routes).resolve(model) -> RouteConfig | None` used identically in Task 4 + tests. `load_config(path|None) -> Config` / `default_config()` / `ConfigError` consistent across Tasks 2/5. `create_app(*, router=, ...)` signature matches all call sites in Task 4 tests + Task 5 `serve`. ✓

**Known on-device-only gaps (by design):** `serve`'s uvicorn/ZeusMeter path and the multi-backend on-device routing (Task 5 Step 5) need real backends; everything else is hermetic via `FakeMeter` + ASGI fake upstreams + `CliRunner`.
