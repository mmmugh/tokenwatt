# TokenWatt M0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A frictionless, end-to-end TokenWatt core: a thin OpenAI-compatible proxy that forwards `/v1/chat/completions` to one local backend, brackets each request with real per-rail SoC energy (zeus), subtracts an idle baseline, prices it at a flat `$/kWh`, and writes a per-request ledger you can `report` on — installable via `uv` and runnable with no config and no sudo.

**Architecture:** A Starlette ASGI app receives a request, snapshots energy via an injected `EnergyMeter`, forwards the body byte-exact to the upstream with `httpx` streaming, tees the response to count tokens, closes the energy window when the stream ends, computes marginal cost, and inserts a ledger row. Every external dependency (the energy meter, the upstream HTTP client) is injected, so the whole pipeline is unit-testable on any machine with a `FakeMeter` and an in-process fake upstream; only `ZeusMeter` requires real Apple Silicon and is verified on-device against a wall meter.

**Tech Stack:** Python 3.12, `uv`, Starlette + `uvicorn`, `httpx`, `typer` (CLI), `tiktoken` (self-count fallback), `zeus-apple-silicon` (energy), `sqlite3` (stdlib), `pytest` + `pytest-asyncio`.

## Global Constraints

- **Python ≥ 3.12** (matches the user's `mlx-env`; `~/mlx-env/bin/python3` is 3.12.13).
- **No sudo at runtime** — energy comes from `zeus-apple-silicon` (IOReport, no root). Never shell out to `powermetrics` in M0.
- **Byte-exact streaming** — forward upstream response bytes UNCHANGED; never parse-then-reserialize SSE. Tee a copy for counting.
- **Honesty labels** — a number is never stamped `calibrated` in M0 (calibration is M2); energy confidence is `estimated` when uncalibrated, `energy-only` when token usage is unavailable. If no `$/kWh` is configured, cost is `None` and labeled `estimated`.
- **Marginal is the hero** — the M0 cost number is marginal (window energy − idle baseline) × rate. No `total`/`amortized` in M0.
- **Single backend, generative only** — M0 forwards to ONE upstream (default `http://127.0.0.1:8080`, override `--upstream`); routing config, vision/embeddings types, and multi-backend are M1.
- **Default port 7000**, host `127.0.0.1`.
- **Version** is read from the repo `VERSION` file (auto-incremented patch per commit by `.githooks/pre-commit`); do not hardcode it.
- **Commit style** — Conventional Commit prefixes (`feat:`, `test:`, `chore:`, `docs:`). Normal commits trigger the version bump; that is expected.
- **Deferred ledger column** — `tok_confidence` (spec §10) is deferred to M1; M0 records `tok_source` (`backend`/`self-count`/`none`), which already encodes token-count confidence.

---

## File Structure

```
pyproject.toml                 # packaging, deps, console entry, dynamic version from VERSION
src/tokenwatt/__init__.py      # package marker + __version__
src/tokenwatt/meter.py         # EnergyByRail, EnergyMeter protocol, ZeusMeter, FakeMeter
src/tokenwatt/idle.py          # IdleBaseline sampler (per-rail idle power)
src/tokenwatt/usage.py         # TokenUsage, OpenAI extractor + streaming self-count
src/tokenwatt/rate.py          # FlatRate.price(kwh) -> $ | None
src/tokenwatt/cost.py          # marginal_kwh(window, idle), cost helpers
src/tokenwatt/ledger.py        # sqlite schema, insert(), rollup queries
src/tokenwatt/proxy.py         # Starlette app factory: forward + bracket + record
src/tokenwatt/cli.py           # typer: `serve`, `report`
tests/test_meter.py
tests/test_idle.py
tests/test_usage.py
tests/test_cost.py
tests/test_ledger.py
tests/test_proxy.py            # integration: fake upstream + FakeMeter
tests/conftest.py              # shared fixtures (tmp ledger, FakeMeter, fake upstream app)
```

One responsibility per file. `proxy.py` is the only place that wires the others together; everything it depends on is passed in.

---

### Task 1: Project scaffold, packaging, version wiring

**Files:**
- Create: `pyproject.toml`
- Create: `src/tokenwatt/__init__.py`
- Create: `src/tokenwatt/cli.py`
- Create: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: console command `tokenwatt` → `tokenwatt.cli:main` (a `typer` app). `tokenwatt --version` prints the `VERSION` file contents. `tokenwatt.__version__: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_smoke.py
import subprocess, sys, pathlib

def test_version_matches_version_file():
    root = pathlib.Path(__file__).resolve().parents[1]
    expected = (root / "VERSION").read_text().strip()
    out = subprocess.run(
        [sys.executable, "-m", "tokenwatt", "--version"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert expected in out.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py -v`
Expected: FAIL — `No module named tokenwatt` (package not built yet).

- [ ] **Step 3: Create the package and packaging**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tokenwatt"
description = "Know what your local LLM inference actually costs in electricity (Apple Silicon)."
readme = "README.md"
requires-python = ">=3.12"
dynamic = ["version"]
dependencies = [
    "starlette>=1.0,<2",
    "uvicorn>=0.30",
    "httpx>=0.27",
    "typer>=0.12",
    "tiktoken>=0.7",
    "zeus-apple-silicon>=1.0",
]

[project.scripts]
tokenwatt = "tokenwatt.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.setuptools.dynamic]
version = { file = "VERSION" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```python
# src/tokenwatt/__init__.py
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tokenwatt")
except PackageNotFoundError:  # running from source without install
    import pathlib
    __version__ = (pathlib.Path(__file__).resolve().parents[2] / "VERSION").read_text().strip()
```

```python
# src/tokenwatt/cli.py
import typer
from tokenwatt import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _version_cb(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True),
):
    """TokenWatt — local-inference electricity cost meter."""


def main():
    app()
```

Also create the `python -m tokenwatt` entrypoint and an empty test fixtures file:

```python
# src/tokenwatt/__main__.py
from tokenwatt.cli import main

main()
```

Create an empty `tests/conftest.py` for now.

- [ ] **Step 4: Install editable and run the test**

Run: `uv pip install -e ".[dev]" && uv run pytest tests/test_cli_smoke.py -v`
Expected: PASS. (`python -m tokenwatt` resolves via `src/tokenwatt/__main__.py`, created in Step 3.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/tokenwatt tests/test_cli_smoke.py tests/conftest.py
git commit -m "feat: scaffold tokenwatt package with version-from-VERSION CLI"
```

---

### Task 2: Energy meter abstraction (`EnergyByRail`, `EnergyMeter`, `ZeusMeter`, `FakeMeter`)

**Files:**
- Create: `src/tokenwatt/meter.py`
- Test: `tests/test_meter.py`

**Interfaces:**
- Produces:
  - `EnergyByRail` — frozen dataclass; field `joules: dict[str, float]` (per-rail joules, only present rails); property `total_j: float`; `__sub__(other) -> EnergyByRail` (per-rail clamped-at-zero difference); classmethod `from_zeus(metrics) -> EnergyByRail`.
  - `RAILS: tuple[str, ...] = ("cpu_total", "gpu", "gpu_sram", "dram", "ane")`
  - `EnergyMeter` Protocol: `begin(label: str) -> None`, `end(label: str) -> EnergyByRail`, `cumulative() -> EnergyByRail`.
  - `ZeusMeter()` — real implementation over `zeus_apple_silicon.AppleEnergyMonitor`.
  - `FakeMeter(windows: dict[str, EnergyByRail] | None = None, cumulative_step: EnergyByRail | None = None)` — deterministic test double; `end(label)` returns the configured window (default 1.0 J on each rail); `cumulative()` advances by `cumulative_step` each call (default 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meter.py
from tokenwatt.meter import EnergyByRail, FakeMeter, RAILS


def test_energybyrail_total_and_sub():
    a = EnergyByRail({"gpu": 5.0, "dram": 3.0})
    b = EnergyByRail({"gpu": 2.0, "dram": 4.0})
    assert a.total_j == 8.0
    diff = a - b
    assert diff.joules["gpu"] == 3.0
    assert diff.joules["dram"] == 0.0          # clamped at zero, never negative


def test_from_zeus_converts_mj_to_j_and_skips_none():
    class M:  # mimics AppleEnergyMetrics
        cpu_total_mj = 1000
        gpu_mj = 2000
        gpu_sram_mj = None
        dram_mj = 500
        ane_mj = None
    e = EnergyByRail.from_zeus(M())
    assert e.joules == {"cpu_total": 1.0, "gpu": 2.0, "dram": 0.5}
    assert e.total_j == 3.5


def test_fake_meter_window_default_one_joule_per_rail():
    m = FakeMeter()
    m.begin("r1")
    w = m.end("r1")
    assert w.total_j == float(len(RAILS))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_meter.py -v`
Expected: FAIL — `No module named tokenwatt.meter`.

- [ ] **Step 3: Implement `meter.py`**

```python
# src/tokenwatt/meter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

RAILS: tuple[str, ...] = ("cpu_total", "gpu", "gpu_sram", "dram", "ane")

# AppleEnergyMetrics attribute -> our rail name. Values are millijoules.
_ZEUS_FIELDS = {
    "cpu_total_mj": "cpu_total",
    "gpu_mj": "gpu",
    "gpu_sram_mj": "gpu_sram",
    "dram_mj": "dram",
    "ane_mj": "ane",
}


@dataclass(frozen=True)
class EnergyByRail:
    joules: dict[str, float]

    @property
    def total_j(self) -> float:
        return sum(self.joules.values())

    def __sub__(self, other: "EnergyByRail") -> "EnergyByRail":
        rails = set(self.joules) | set(other.joules)
        return EnergyByRail({
            r: max(0.0, self.joules.get(r, 0.0) - other.joules.get(r, 0.0))
            for r in rails
        })

    @classmethod
    def from_zeus(cls, metrics) -> "EnergyByRail":
        out: dict[str, float] = {}
        for attr, rail in _ZEUS_FIELDS.items():
            val = getattr(metrics, attr, None)
            if val is not None:
                out[rail] = val / 1000.0   # mJ -> J
        return cls(out)


@runtime_checkable
class EnergyMeter(Protocol):
    def begin(self, label: str) -> None: ...
    def end(self, label: str) -> EnergyByRail: ...
    def cumulative(self) -> EnergyByRail: ...


class ZeusMeter:
    """Real meter over Apple Silicon IOReport. No sudo. Requires Apple Silicon."""

    def __init__(self) -> None:
        from zeus_apple_silicon import AppleEnergyMonitor
        self._mon = AppleEnergyMonitor()

    def begin(self, label: str) -> None:
        self._mon.begin_window(label)

    def end(self, label: str) -> EnergyByRail:
        return EnergyByRail.from_zeus(self._mon.end_window(label))

    def cumulative(self) -> EnergyByRail:
        return EnergyByRail.from_zeus(self._mon.get_cumulative_energy())


class FakeMeter:
    """Deterministic test double. No hardware."""

    def __init__(self, windows: dict[str, EnergyByRail] | None = None,
                 cumulative_step: EnergyByRail | None = None) -> None:
        self._windows = windows or {}
        self._default = EnergyByRail({r: 1.0 for r in RAILS})
        self._step = cumulative_step or EnergyByRail({})
        self._accum = EnergyByRail({})
        self._open: set[str] = set()

    def begin(self, label: str) -> None:
        self._open.add(label)

    def end(self, label: str) -> EnergyByRail:
        self._open.discard(label)
        return self._windows.get(label, self._default)

    def cumulative(self) -> EnergyByRail:
        rails = set(self._accum.joules) | set(self._step.joules)
        self._accum = EnergyByRail({
            r: self._accum.joules.get(r, 0.0) + self._step.joules.get(r, 0.0)
            for r in rails
        })
        return self._accum
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_meter.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/meter.py tests/test_meter.py
git commit -m "feat: per-rail energy meter abstraction with zeus + fake implementations"
```

---

### Task 3: Idle baseline

**Files:**
- Create: `src/tokenwatt/idle.py`
- Test: `tests/test_idle.py`

**Interfaces:**
- Consumes: `EnergyByRail`, `EnergyMeter` (Task 2).
- Produces: `IdleBaseline(meter: EnergyMeter)` with `sample() -> None` (call when idle; records a cumulative reading + monotonic time and updates per-rail watts from the delta) and `energy_over(seconds: float) -> EnergyByRail` (per-rail idle energy for a window of that duration; zero before two samples exist).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_idle.py
from tokenwatt.meter import EnergyByRail, FakeMeter
from tokenwatt.idle import IdleBaseline


def test_idle_power_from_two_samples(monkeypatch):
    # cumulative advances 2 J on gpu each sample; we control the clock.
    meter = FakeMeter(cumulative_step=EnergyByRail({"gpu": 2.0}))
    clock = {"t": 0.0}
    import tokenwatt.idle as idle_mod
    monkeypatch.setattr(idle_mod.time, "monotonic", lambda: clock["t"])

    base = IdleBaseline(meter)
    base.sample()                 # t=0, cumulative gpu=2.0
    clock["t"] = 1.0
    base.sample()                 # t=1, cumulative gpu=4.0 -> 2.0 J over 1.0 s -> 2.0 W
    e = base.energy_over(3.0)     # 2.0 W * 3 s = 6.0 J
    assert e.joules["gpu"] == 6.0


def test_energy_over_zero_before_two_samples():
    base = IdleBaseline(FakeMeter())
    assert base.energy_over(5.0).total_j == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_idle.py -v`
Expected: FAIL — `No module named tokenwatt.idle`.

- [ ] **Step 3: Implement `idle.py`**

```python
# src/tokenwatt/idle.py
from __future__ import annotations
import time
from tokenwatt.meter import EnergyByRail, EnergyMeter


class IdleBaseline:
    """Rolling per-rail idle power (watts) measured from cumulative energy deltas."""

    def __init__(self, meter: EnergyMeter) -> None:
        self._meter = meter
        self._last: EnergyByRail | None = None
        self._last_t: float | None = None
        self._watts: dict[str, float] = {}

    def sample(self) -> None:
        now = time.monotonic()
        cur = self._meter.cumulative()
        if self._last is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                delta = cur - self._last
                self._watts = {r: j / dt for r, j in delta.joules.items()}
        self._last, self._last_t = cur, now

    def energy_over(self, seconds: float) -> EnergyByRail:
        return EnergyByRail({r: w * seconds for r, w in self._watts.items()})
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_idle.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/idle.py tests/test_idle.py
git commit -m "feat: rolling idle-baseline power sampler"
```

---

### Task 4: Token usage (OpenAI extract + streaming self-count)

**Files:**
- Create: `src/tokenwatt/usage.py`
- Test: `tests/test_usage.py`

**Interfaces:**
- Produces:
  - `TokenUsage` dataclass: `input: int | None`, `output: int | None`, `cached: int | None`, `source: str`, `confidence: str`.
  - `usage_from_response_json(body: dict) -> TokenUsage | None` — reads OpenAI `usage` block; `source="backend"`, `confidence="high"`; returns `None` if no usage block.
  - `SelfCounter(request_body: dict)` — for streaming: `feed(chunk: bytes) -> None` accumulates assistant `content` deltas from SSE; `result() -> TokenUsage` tokenizes input (request messages) and output (accumulated content) with `tiktoken` `cl100k_base`, `source="self-count"`, `confidence="low"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_usage.py
from tokenwatt.usage import usage_from_response_json, SelfCounter


def test_usage_from_backend_response():
    body = {"usage": {"prompt_tokens": 12, "completion_tokens": 34,
                      "prompt_tokens_details": {"cached_tokens": 4}}}
    u = usage_from_response_json(body)
    assert (u.input, u.output, u.cached) == (12, 34, 4)
    assert u.source == "backend" and u.confidence == "high"


def test_usage_from_response_none_when_absent():
    assert usage_from_response_json({"choices": []}) is None


def test_self_counter_counts_streamed_content():
    req = {"messages": [{"role": "user", "content": "hello world"}]}
    sc = SelfCounter(req)
    # two SSE data chunks carrying content deltas, then [DONE]
    sc.feed(b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n')
    sc.feed(b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n')
    sc.feed(b'data: [DONE]\n\n')
    u = sc.result()
    assert u.output >= 1 and u.input >= 1
    assert u.source == "self-count" and u.confidence == "low"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_usage.py -v`
Expected: FAIL — `No module named tokenwatt.usage`.

- [ ] **Step 3: Implement `usage.py`**

```python
# src/tokenwatt/usage.py
from __future__ import annotations
import json
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")  # model-agnostic approximation


@dataclass
class TokenUsage:
    input: int | None
    output: int | None
    cached: int | None
    source: str       # "backend" | "self-count" | "none"
    confidence: str   # "high" | "low" | "energy-only"


def usage_from_response_json(body: dict) -> TokenUsage | None:
    usage = body.get("usage")
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or {}
    return TokenUsage(
        input=usage.get("prompt_tokens"),
        output=usage.get("completion_tokens"),
        cached=details.get("cached_tokens"),
        source="backend",
        confidence="high",
    )


def _count(text: str) -> int:
    return len(_ENC.encode(text or ""))


class SelfCounter:
    """Counts output tokens from streamed SSE content deltas (fallback only)."""

    def __init__(self, request_body: dict) -> None:
        msgs = request_body.get("messages") or []
        self._input = sum(
            _count(m.get("content", "")) for m in msgs if isinstance(m.get("content"), str)
        )
        self._buf = ""
        self._out_text = ""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk.decode("utf-8", errors="ignore")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    self._out_text += delta["content"]

    def result(self) -> TokenUsage:
        return TokenUsage(
            input=self._input,
            output=_count(self._out_text),
            cached=None,
            source="self-count",
            confidence="low",
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_usage.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/usage.py tests/test_usage.py
git commit -m "feat: OpenAI usage extraction + streaming self-count fallback"
```

---

### Task 5: Rate + cost math

**Files:**
- Create: `src/tokenwatt/rate.py`
- Create: `src/tokenwatt/cost.py`
- Test: `tests/test_cost.py`

**Interfaces:**
- Consumes: `EnergyByRail` (Task 2).
- Produces:
  - `FlatRate(usd_per_kwh: float | None)` with `price(kwh: float) -> float | None` (None when rate unset).
  - `marginal_kwh(window: EnergyByRail, idle: EnergyByRail) -> float` — `(window - idle).total_j / 3.6e6`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost.py
from tokenwatt.meter import EnergyByRail
from tokenwatt.rate import FlatRate
from tokenwatt.cost import marginal_kwh


def test_marginal_kwh_subtracts_idle():
    window = EnergyByRail({"gpu": 3_600_000.0})   # 3.6 MJ = 1 kWh
    idle = EnergyByRail({"gpu": 1_800_000.0})     # 0.5 kWh
    assert marginal_kwh(window, idle) == 0.5


def test_flat_rate_prices_and_handles_unset():
    assert FlatRate(0.31).price(0.5) == 0.155
    assert FlatRate(None).price(0.5) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cost.py -v`
Expected: FAIL — `No module named tokenwatt.rate`.

- [ ] **Step 3: Implement `rate.py` and `cost.py`**

```python
# src/tokenwatt/rate.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FlatRate:
    usd_per_kwh: float | None

    def price(self, kwh: float) -> float | None:
        if self.usd_per_kwh is None:
            return None
        return kwh * self.usd_per_kwh
```

```python
# src/tokenwatt/cost.py
from __future__ import annotations
from tokenwatt.meter import EnergyByRail


def marginal_kwh(window: EnergyByRail, idle: EnergyByRail) -> float:
    """Marginal energy (window minus idle baseline) in kWh."""
    return (window - idle).total_j / 3.6e6
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cost.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/rate.py src/tokenwatt/cost.py tests/test_cost.py
git commit -m "feat: flat rate pricing and marginal-kWh cost math"
```

---

### Task 6: Ledger (sqlite)

**Files:**
- Create: `src/tokenwatt/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces:
  - `LedgerRow` dataclass with fields: `ts_start: float`, `ts_end: float`, `model: str`, `e_window_j: float`, `e_idle_j: float`, `e_marginal_j: float`, `kwh_marginal: float`, `rate_usd_kwh: float | None`, `cost_marginal_usd: float | None`, `tok_in: int | None`, `tok_out: int | None`, `tok_source: str`, `energy_confidence: str`.
  - `Ledger(path: str)` with `insert(row: LedgerRow) -> None`, `by_model() -> list[dict]` (per-model count, total kWh, total $, sum tok_out, J/output-token), and `totals(since_epoch: float) -> dict` (count, kWh, $ for rows with `ts_start >= since_epoch`). Creates the schema on construction.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ledger.py
from tokenwatt.ledger import Ledger, LedgerRow


def _row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000):
    return LedgerRow(
        ts_start=100.0, ts_end=101.0, model=model,
        e_window_j=marg_j + 100, e_idle_j=100, e_marginal_j=marg_j,
        kwh_marginal=marg_j / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=cost,
        tok_in=10, tok_out=tok_out, tok_source="backend", energy_confidence="estimated",
    )


def test_insert_and_by_model_rollup(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000))
    led.insert(_row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000))
    rows = led.by_model()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "m1" and r["requests"] == 2
    assert abs(r["total_usd"] - 0.62) < 1e-9
    assert abs(r["j_per_out_token"] - (7_200_000.0 / 2000)) < 1e-6


def test_totals_since(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row())
    t = led.totals(since_epoch=0.0)
    assert t["requests"] == 1 and abs(t["usd"] - 0.31) < 1e-9


def test_cost_is_none_when_no_rate(tmp_path):
    # Honesty contract: with no rate, cost must stay NULL through the rollups,
    # never collapse to $0 (which would read as "free").
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=100.0, ts_end=101.0, model="m1",
        e_window_j=10.0, e_idle_j=1.0, e_marginal_j=9.0,
        kwh_marginal=9.0 / 3.6e6, rate_usd_kwh=None, cost_marginal_usd=None,
        tok_in=1, tok_out=1, tok_source="self-count", energy_confidence="estimated (±15-30%)",
    ))
    assert led.by_model()[0]["total_usd"] is None
    assert led.totals(0.0)["usd"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ledger.py -v`
Expected: FAIL — `No module named tokenwatt.ledger`.

- [ ] **Step 3: Implement `ledger.py`**

```python
# src/tokenwatt/ledger.py
from __future__ import annotations
import sqlite3
from dataclasses import asdict, dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL, ts_end REAL, model TEXT,
    e_window_j REAL, e_idle_j REAL, e_marginal_j REAL,
    kwh_marginal REAL, rate_usd_kwh REAL, cost_marginal_usd REAL,
    tok_in INTEGER, tok_out INTEGER, tok_source TEXT, energy_confidence TEXT
);
"""


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


class Ledger:
    def __init__(self, path: str) -> None:
        self._path = path
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, row: LedgerRow) -> None:
        d = asdict(row)
        cols = ", ".join(d)
        ph = ", ".join("?" for _ in d)
        with self._conn() as c:
            c.execute(f"INSERT INTO requests ({cols}) VALUES ({ph})", tuple(d.values()))

    def by_model(self) -> list[dict]:
        sql = """
        SELECT model,
               COUNT(*)                AS requests,
               COALESCE(SUM(kwh_marginal), 0)      AS total_kwh,
               SUM(cost_marginal_usd)             AS total_usd,
               COALESCE(SUM(tok_out), 0)           AS total_out,
               COALESCE(SUM(e_marginal_j), 0)      AS total_marginal_j
        FROM requests GROUP BY model ORDER BY total_usd DESC
        """
        out = []
        with self._conn() as c:
            for r in c.execute(sql):
                d = dict(r)
                d["j_per_out_token"] = (d["total_marginal_j"] / d["total_out"]) if d["total_out"] else None
                out.append(d)
        return out

    def totals(self, since_epoch: float) -> dict:
        sql = """
        SELECT COUNT(*) AS requests,
               COALESCE(SUM(kwh_marginal), 0)      AS kwh,
               SUM(cost_marginal_usd)             AS usd
        FROM requests WHERE ts_start >= ?
        """
        with self._conn() as c:
            return dict(c.execute(sql, (since_epoch,)).fetchone())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_ledger.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/ledger.py tests/test_ledger.py
git commit -m "feat: sqlite ledger with per-model and period rollups"
```

---

### Task 7: Proxy (forward byte-exact + bracket energy + record)

**Files:**
- Create: `src/tokenwatt/proxy.py`
- Create: `tests/conftest.py` (replace the empty stub)
- Test: `tests/test_proxy.py`

**Interfaces:**
- Consumes: `EnergyMeter`, `EnergyByRail` (T2), `IdleBaseline` (T3), `usage_from_response_json` + `SelfCounter` (T4), `FlatRate` + `marginal_kwh` (T5), `Ledger` + `LedgerRow` (T6).
- Produces: `create_app(*, upstream: str, meter: EnergyMeter, idle: IdleBaseline, ledger: Ledger, rate: FlatRate, client: httpx.AsyncClient, lifespan=None) -> Starlette`. Forwards `POST /v1/{path:path}` to `f"{upstream}/v1/{path}"`, streaming the response body unchanged; after the body is fully sent it closes the energy window, computes marginal cost, and inserts one `LedgerRow` (model taken from the request JSON `model` field, default `"unknown"`).

- [ ] **Step 1: Write the failing integration test**

```python
# tests/conftest.py
import json
import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route


@pytest.fixture
def fake_upstream_streaming():
    """An OpenAI-ish upstream that streams two content chunks then [DONE], no usage."""
    async def chat(request):
        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.fixture
def fake_upstream_json():
    """A non-streaming upstream that returns a usage block."""
    async def chat(request):
        return JSONResponse({
            "choices": [{"message": {"content": "Hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])
```

```python
# tests/test_proxy.py
import json
import httpx
import pytest

from tokenwatt.proxy import create_app
from tokenwatt.meter import EnergyByRail, FakeMeter
from tokenwatt.idle import IdleBaseline
from tokenwatt.rate import FlatRate
from tokenwatt.ledger import Ledger


def _client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://up")


async def test_streaming_passthrough_is_byte_exact_and_records(tmp_path, fake_upstream_streaming):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    meter = FakeMeter(windows={"x": EnergyByRail({"gpu": 3_600_000.0})})  # 1 kWh window
    # Force the request id label so FakeMeter returns our window:
    app = create_app(
        upstream="http://up", meter=meter, idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(0.31),
        client=_client_for(fake_upstream_streaming),
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
        upstream="http://up", meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(0.31),
        client=_client_for(fake_upstream_json),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions",
                         json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        assert r.json()["usage"]["completion_tokens"] == 2
    with ledger._conn() as conn:
        row = conn.execute("SELECT tok_out, tok_source FROM requests").fetchone()
    assert row["tok_out"] == 2 and row["tok_source"] == "backend"


async def test_cost_none_when_rate_unset(tmp_path, fake_upstream_json):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    app = create_app(
        upstream="http://up", meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
        ledger=ledger, rate=FlatRate(None),
        client=_client_for(fake_upstream_json),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://tw") as c:
        await c.post("/v1/chat/completions",
                     json={"model": "m1", "messages": [{"role": "user", "content": "hi"}]})
    with ledger._conn() as conn:
        row = conn.execute("SELECT cost_marginal_usd FROM requests").fetchone()
    assert row["cost_marginal_usd"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_proxy.py -v`
Expected: FAIL — `No module named tokenwatt.proxy`.

- [ ] **Step 3: Implement `proxy.py`**

```python
# src/tokenwatt/proxy.py
from __future__ import annotations
import json
import time
import uuid
from typing import Callable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from tokenwatt.meter import EnergyMeter
from tokenwatt.idle import IdleBaseline
from tokenwatt.rate import FlatRate
from tokenwatt.cost import marginal_kwh
from tokenwatt.usage import usage_from_response_json, SelfCounter, TokenUsage
from tokenwatt.ledger import Ledger, LedgerRow

# Hop-by-hop headers that must not be forwarded.
_DROP = {"content-length", "transfer-encoding", "connection", "host"}


def create_app(*, upstream: str, meter: EnergyMeter, idle: IdleBaseline,
               ledger: Ledger, rate: FlatRate, client: httpx.AsyncClient,
               _label_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
               lifespan=None) -> Starlette:

    async def forward(request: Request) -> Response:
        path = request.path_params["path"]
        raw = await request.body()
        try:
            req_json = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            req_json = {}
        model = req_json.get("model", "unknown")
        is_stream = bool(req_json.get("stream"))

        fwd_headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _DROP]

        label = _label_factory()
        meter.begin(label)
        t0 = time.monotonic()
        ts_start = time.time()

        up_req = client.build_request(
            "POST", f"{upstream}/v1/{path}", content=raw, headers=fwd_headers
        )
        up_resp = await client.send(up_req, stream=True)

        counter = SelfCounter(req_json) if is_stream else None
        captured = bytearray() if not is_stream else None

        def _finalize(usage: TokenUsage) -> None:
            window = meter.end(label)
            dt = time.monotonic() - t0
            idle_e = idle.energy_over(dt)
            kwh = marginal_kwh(window, idle_e)
            cost = rate.price(kwh)
            ledger.insert(LedgerRow(
                ts_start=ts_start, ts_end=time.time(), model=model,
                e_window_j=window.total_j, e_idle_j=idle_e.total_j,
                e_marginal_j=(window - idle_e).total_j, kwh_marginal=kwh,
                rate_usd_kwh=rate.usd_per_kwh, cost_marginal_usd=cost,
                tok_in=usage.input if usage else None,
                tok_out=usage.output if usage else None,
                tok_source=usage.source if usage else "none",
                energy_confidence="estimated (±15-30%)" if usage and usage.source != "none" else "energy-only",
            ))

        resp_headers = [(k, v) for k, v in up_resp.headers.items() if k.lower() not in _DROP]

        if is_stream:
            async def body_iter():
                try:
                    async for chunk in up_resp.aiter_raw():
                        counter.feed(chunk)
                        yield chunk
                finally:
                    await up_resp.aclose()
                    _finalize(counter.result())
            return StreamingResponse(body_iter(), status_code=up_resp.status_code,
                                     headers=dict(resp_headers))
        else:
            async for chunk in up_resp.aiter_raw():
                captured.extend(chunk)
            await up_resp.aclose()
            body = bytes(captured)
            try:
                usage = usage_from_response_json(json.loads(body))
            except json.JSONDecodeError:
                usage = None
            _finalize(usage or TokenUsage(None, None, None, "none", "energy-only"))
            return Response(content=body, status_code=up_resp.status_code,
                            headers=dict(resp_headers))

    return Starlette(routes=[Route("/v1/{path:path}", forward, methods=["POST"])], lifespan=lifespan)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_proxy.py -v`
Expected: PASS (2 tests). If `aiter_raw` over `ASGITransport` buffers, the byte-exact assertion still holds because we yield exactly the upstream frames.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS (all tests from Tasks 1–7).

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/proxy.py tests/test_proxy.py tests/conftest.py
git commit -m "feat: byte-exact forwarding proxy that brackets energy and records the ledger"
```

---

### Task 8: CLI `serve` + `report`, zero-config, and on-device verification

**Files:**
- Modify: `src/tokenwatt/cli.py`
- Create: `README.md` (replace the design-status stub with the install + run hero)
- Test: `tests/test_report_render.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `tokenwatt serve [--upstream URL] [--port 7000] [--rate 0.31] [--ledger PATH]` — boots `uvicorn` on the Starlette app with a `ZeusMeter`, an `IdleBaseline` sampled on a background timer, and a `Ledger`. `tokenwatt report [--ledger PATH]` — prints today + month totals and a per-model table. `render_report(ledger: Ledger, now: float) -> str` is the pure, testable formatter.

- [ ] **Step 1: Write the failing test for the pure renderer**

```python
# tests/test_report_render.py
from tokenwatt.ledger import Ledger, LedgerRow
from tokenwatt.cli import render_report


def test_render_report_shows_model_and_dollars(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=3_600_100.0, e_idle_j=100.0, e_marginal_j=3_600_000.0,
        kwh_marginal=1.0, rate_usd_kwh=0.31, cost_marginal_usd=0.31,
        tok_in=10, tok_out=1000, tok_source="backend", energy_confidence="estimated",
    ))
    text = render_report(led, now=1002.0)
    assert "m1" in text
    assert "$0.31" in text
    assert "estimated" in text.lower()   # honesty banner


def test_render_report_dashes_when_no_rate(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=10.0, e_idle_j=1.0, e_marginal_j=9.0,
        kwh_marginal=9.0 / 3.6e6, rate_usd_kwh=None, cost_marginal_usd=None,
        tok_in=1, tok_out=1, tok_source="self-count", energy_confidence="estimated (±15-30%)",
    ))
    text = render_report(led, now=1002.0)
    assert "$0.0000" not in text   # never render an unpriced row as free
    assert "—" in text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_report_render.py -v`
Expected: FAIL — `cannot import name 'render_report'`.

- [ ] **Step 3: Implement `serve`, `render_report`, and `report` in `cli.py`**

```python
# add to src/tokenwatt/cli.py
import asyncio
import time
from typing import Optional

import typer
import httpx

from tokenwatt.ledger import Ledger
from tokenwatt.rate import FlatRate
from tokenwatt.idle import IdleBaseline


def _usd(x: float | None) -> str:
    return f"${x:.4f}" if x is not None else "—"


def render_report(ledger: Ledger, now: float) -> str:
    day = ledger.totals(now - 86_400)
    month = ledger.totals(now - 30 * 86_400)
    lines = [
        "TokenWatt — electricity cost of local inference",
        "  (numbers are ESTIMATED until you calibrate against a wall meter)",
        f"  last 24h : {day['requests']:>6} req   {day['kwh']:.4f} kWh   {_usd(day['usd'])}",
        f"  last 30d : {month['requests']:>6} req   {month['kwh']:.4f} kWh   {_usd(month['usd'])}",
        "",
        f"  {'model':<16}{'req':>8}{'kWh':>12}{'$':>10}{'J/out-tok':>12}",
    ]
    for r in ledger.by_model():
        jpt = f"{r['j_per_out_token']:.3f}" if r["j_per_out_token"] is not None else "-"
        lines.append(
            f"  {r['model']:<16}{r['requests']:>8}{r['total_kwh']:>12.4f}"
            f"{_usd(r['total_usd']):>10}{jpt:>12}"
        )
    return "\n".join(lines)


@app.command()
def report(ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger")):
    """Show today/month electricity cost and a per-model breakdown."""
    import os
    path = os.path.expanduser(ledger)
    typer.echo(render_report(Ledger(path), now=time.time()))


@app.command()
def serve(
    upstream: str = typer.Option("http://127.0.0.1:8080", "--upstream"),
    port: int = typer.Option(7000, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    rate: Optional[float] = typer.Option(None, "--rate", help="flat $/kWh; omit to label costs 'estimated'"),
    ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger"),
):
    """Run the measuring proxy (no sudo)."""
    import os
    from contextlib import asynccontextmanager
    import uvicorn
    from tokenwatt.meter import ZeusMeter
    from tokenwatt.proxy import create_app

    os.makedirs(os.path.dirname(os.path.expanduser(ledger)), exist_ok=True)
    led = Ledger(os.path.expanduser(ledger))
    meter = ZeusMeter()
    idle = IdleBaseline(meter)
    client = httpx.AsyncClient(timeout=None)

    # Starlette 1.x removed on_event; the idle sampler runs inside an ASGI lifespan.
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

    app_asgi = create_app(upstream=upstream, meter=meter, idle=idle,
                          ledger=led, rate=FlatRate(rate), client=client,
                          lifespan=lifespan)

    typer.echo(f"TokenWatt proxy on http://{host}:{port}  ->  {upstream}  (no sudo)")
    uvicorn.run(app_asgi, host=host, port=port, log_level="warning")
```

- [ ] **Step 4: Run the renderer test and full suite**

Run: `uv run pytest tests/test_report_render.py -v && uv run pytest -v`
Expected: PASS (all).

- [ ] **Step 5: Write the README hero (install + bare run + no sudo)**

```markdown
# TokenWatt

**Know what your local LLM inference actually costs in electricity — Apple Silicon, no sudo.**

```bash
uv tool install tokenwatt          # or: uvx tokenwatt
tokenwatt serve --upstream http://127.0.0.1:8080 --rate 0.31
# point your OpenAI client at http://127.0.0.1:7000
tokenwatt report
```

Try it without installing:
`uvx --from git+https://github.com/<you>/tokenwatt@v0.1.x tokenwatt serve`

No password prompt, ever — energy is read via Apple's IOReport (the same sudoless path
macmon/mactop use). See the design spec in `docs/design/specs/`.
```

- [ ] **Step 6: On-device verification (M3 Ultra) — the real acceptance test**

```bash
# 1. start a real backend (one of the user's), e.g.:
mlx-openai-server --model <some-mlx-model> --port 8080   # in another shell
# 2. run the proxy:
uv run tokenwatt serve --upstream http://127.0.0.1:8080 --rate 0.31
# 3. send a real request through the proxy:
curl -s http://127.0.0.1:7000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m1","messages":[{"role":"user","content":"Write a haiku about watts."}]}' >/dev/null
# 4. read the ledger:
uv run tokenwatt report
```

Expected: `report` shows 1 request under model `m1`, a non-zero kWh, and a `$` figure ≈ kWh × 0.31. Sanity-check the kWh against a wall-meter reading taken across a longer sustained generation (energy should be within tens of percent uncalibrated — calibration tightens this in M2).

- [ ] **Step 7: Commit**

```bash
git add src/tokenwatt/cli.py README.md tests/test_report_render.py
git commit -m "feat: serve + report CLI with zero-sudo ZeusMeter and on-device verification"
```

---

## Self-Review

**1. Spec coverage (M0 slice of the spec):**
- Thin OpenAI passthrough, one backend → Task 7 (`create_app`), Task 8 (`serve --upstream`). ✓
- zeus per-rail window → Task 2 (`ZeusMeter`), Task 7 (begin/end around forward). ✓
- Idle-baseline subtraction → Task 3, used in Task 7 `_finalize`. ✓
- sqlite ledger + per-model rollups → Task 6. ✓
- Flat-rate marginal `$` → Task 5, Task 7. ✓
- `report` CLI → Task 8. ✓
- Backend-usage extraction + self-count fallback → Task 4, wired in Task 7. ✓
- No sudo → `ZeusMeter` uses IOReport only; asserted in README hero + `serve` banner. ✓
- Byte-exact streaming → Task 7 test asserts identical SSE bytes. ✓
- Honesty labels (`estimated (±15-30%)` / `energy-only`; cost `None` when rate unset → kept NULL through Task 6 rollups → rendered `—`, never `$0.0000`) → Task 6 SQL, Task 7 `_finalize`, Task 8 renderer + tests. ✓
- Version from `VERSION` → Task 1. ✓
- Deferred correctly (NOT in M0): YAML config/routing, vision/embeddings, calibration, TOU, Ollama/Anthropic ingress, `wrap` card, menu-bar. ✓

**2. Placeholder scan:** no TBD/TODO; every code step is complete and runnable; commands have expected output. ✓

**3. Type consistency:** `EnergyByRail.total_j`, `.joules`, `__sub__`, `from_zeus` used identically in Tasks 2/3/5/7. `TokenUsage(input, output, cached, source, confidence)` constructed the same in Tasks 4/7. `LedgerRow` field set identical in Tasks 6/7/8. `create_app(*, upstream, meter, idle, ledger, rate, client, _label_factory)` signature matches its call sites in Task 8 and both proxy tests. ✓

**Known on-device-only gaps (by design):** `ZeusMeter` and the `serve` uvicorn path are not unit-tested (they need Apple Silicon + a live backend); they are covered by the Task 8 Step 6 manual acceptance test. Every other unit is hermetic via `FakeMeter` + ASGI fake upstream.
