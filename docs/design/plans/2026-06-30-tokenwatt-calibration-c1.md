# TokenWatt Calibration C1 — Campaign Runner + Synchronized Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive a standardized text load battery (`idle · prefill · decode · balanced`) against an inference upstream, bracket each cell with synchronized rail (zeus) + wall (Shelly) energy, subtract a measured idle baseline from **both** sides, and persist the resulting marginal `(cell, model, Δt, E_rail[], E_wall)` samples for the C2 fit. No fitting, no calibration label — trustworthy sample data only.

**Architecture:** A new `battery.py` defines the load cells (standardized prompts, shipped in-package) and the inference `LoadClient` (an `HttpLoadClient` that POSTs OpenAI chat completions to the upstream + a `FakeLoadClient` test double). `campaign.py` grows the measurement engine on top of C0's `probe()`: `measure_idle` (idle rail+wall power), `run_cell` (sustained-load bracket + marginal math), `run_campaign` (idle → cells × passes → samples), and JSON persistence. `cli.py` gets `tokenwatt calibrate campaign`. Everything is dependency-injected so the whole path runs deterministically on `FakeMeter` + `FakeMeterSource` + `FakeLoadClient` + an injected clock; the real campaign is an on-device step.

**Tech Stack:** Python ≥3.10, `httpx` (sync `Client`, already a dependency), `typer`, `pydantic`, `pytest`. No new dependencies (the fit's numeric libs arrive in C2, not here).

## Global Constraints

- **Python floor:** `requires-python>=3.10`; runs on 3.10–3.14 (CI matrix).
- **Honesty contract (still binding in C1):** the campaign collects **raw marginal samples** — it performs NO fit and must never emit a `calibrated` label, a confidence band, or a fitted coefficient. The persisted artifact is *sample data*, not a profile. (Fit/tier/profile are C2+.)
- **Marginal convention, both sides:** every non-idle cell's rail energy AND wall energy are idle-subtracted — `E_rail_marg[r] = ΔE_rail[r] − idle_rail_w[r]·Δt`, `E_wall_marg = ΔWh·3600 − idle_wall_w·Δt` — matching the proxy's per-request `e_marginal_j`. Clamp each marginal at `≥ 0`.
- **Read the plug's own counter:** wall energy is the delta of the `MeterSource` accumulated Wh, converted Wh→J via ×3600 — never integrated from instantaneous power.
- **Apple-Silicon gated, fail-soft:** `ZeusMeter` constructs only on the Mac being calibrated; off-platform `run_campaign` degrades with a clear `(None, reason)`, never crashes. Fakes run anywhere.
- **Cell duration is cadence-bound:** the plug's measured accumulator cadence on the M3 Ultra is **~66 s**, so the default cell duration is **300 s (~4.5×)** and default **passes = 2** — a cell window must comfortably exceed the cadence or its wall Δ is quantization noise.
- **Non-streaming load:** load requests set `stream=false` + `max_tokens`, so `usage.{prompt,completion}_tokens` is read straight from the response body.
- **Tokens are honest-or-unknown, never a fake 0:** an upstream that omits/nulls `usage` yields `ChatResult(tok_in=None, tok_out=None)` — a cell with any unknown request records `tok_in/tok_out = None` (rendered distinctly, never `0`), matching the project's "never a fabricated number" contract. Tokens are **bookkeeping/validity only** — the fit uses energy. Every cell ALSO records a `requests` count (usage-independent) as the always-available proof that sustained load ran.
- **Deterministic CI:** the whole campaign path runs under `FakeMeter` + `FakeMeterSource` + `FakeLoadClient` + an injected clock — no network, no hardware, no real wall-clock sleeps in tests.
- **`VERSION` auto-bumps** its patch every commit via `.githooks/pre-commit` — expected; let it run. Never `--no-verify`.
- **Leak-safety:** no numeric/private IPs in code, tests, or docs (a pre-commit leak-scan blocks them); test hosts use `"h"`/`shelly.local`.
- **Match existing style:** `from __future__ import annotations`; dataclasses + focused functions like `meter.py`/`campaign.py`; DI seams like `run_probe`/`doctor.run`.

---

## File Structure

- **Create `src/tokenwatt/battery.py`** — `LoadCell`; standardized prompt builders; `text_cells()` (the Core-4 load cells, sans idle); `ChatResult`; `LoadClient` protocol; `HttpLoadClient` (OpenAI chat POST); `FakeLoadClient`. Single responsibility: *what load to generate and how to send it*.
- **Modify `src/tokenwatt/campaign.py`** — add `IdleRates`, `CellSample`, `measure_idle`, `run_cell`, `CampaignResult`, `run_campaign`, `campaign_to_dict`, `write_campaign`. Single responsibility: *bracket, measure, capture, persist* (grows from C0's probe on the same "harness orchestration" theme).
- **Modify `src/tokenwatt/cli.py`** — add the `calibrate campaign` command.
- **Create `tests/test_battery.py`, `tests/test_campaign_run.py`**; extend `tests/test_cli_smoke.py`.

**Deferred out of C1 (do not build here):** `embeddings`/`vision` cells (the user chose Core-4 text; add when an embed/vision route is calibrated — C4 wizard), the fit / `calibration.py` (C2), `machine_id` detection + profile registry (C2), runtime `Cal` + `calib_*` ledger columns (C3), the interactive `calibrate run` wizard + config-route selection + `≥2`-pass repeatability *reporting* UX (C4), per-rail gating (C5). C1 exposes a **non-interactive** `calibrate campaign --upstream --model` that C4 will wrap.

**Folded-in C0 review carry-overs addressed here:** `run_campaign` gets a general `make_source`/`make_load` DI seam (not Shelly-locked); the sustained-load loop guards a mid-window meter/HTTP read failure (returns a clean reason, doesn't traceback).

---

### Task 1: Load cells + standardized prompts (`battery.py`)

**Files:**
- Create: `src/tokenwatt/battery.py`
- Test: `tests/test_battery.py`

**Interfaces:**
- Produces:
  - `LoadCell(name: str, prompt: str, max_tokens: int)` — frozen dataclass; one chat load cell.
  - `text_cells() -> list[LoadCell]` — the three **load** cells `prefill`, `decode`, `balanced` (idle is not a request; it's measured separately). Deterministic; prompts shipped in-package.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_battery.py
from tokenwatt.battery import LoadCell, text_cells


def test_text_cells_are_the_core_three_load_cells():
    cells = text_cells()
    assert [c.name for c in cells] == ["prefill", "decode", "balanced"]
    assert all(isinstance(c, LoadCell) for c in cells)


def test_prefill_is_long_prompt_short_generation():
    # prefill must be compute-bound: a large prompt, a tiny generation cap
    prefill = next(c for c in text_cells() if c.name == "prefill")
    assert prefill.max_tokens <= 16
    assert len(prefill.prompt) > 4000          # ~thousands of tokens of context


def test_decode_is_short_prompt_long_generation():
    # decode must be memory-bound: a small prompt, a large generation cap
    decode = next(c for c in text_cells() if c.name == "decode")
    assert decode.max_tokens >= 512
    assert len(decode.prompt) < 500


def test_balanced_is_between_the_two():
    balanced = next(c for c in text_cells() if c.name == "balanced")
    assert 64 <= balanced.max_tokens <= 512


def test_cells_are_deterministic():
    # standardized battery: identical across runs (version-stamped into C2's profile)
    assert [(c.name, c.prompt, c.max_tokens) for c in text_cells()] == \
           [(c.name, c.prompt, c.max_tokens) for c in text_cells()]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_battery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.battery'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/battery.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadCell:
    """One chat load cell: a standardized prompt + generation cap that stresses
    the SoC a particular way. Shipped in-package so the battery is identical
    across machines and runs (version-stamped into the C2 profile)."""
    name: str
    prompt: str
    max_tokens: int


# A neutral paragraph repeated to a few thousand tokens of context. Deterministic
# and content-free on purpose — we are exercising the machine, not the model.
_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the calibration harness "
    "measures the electricity it takes to think about foxes and dogs and energy. "
)


def _prefill_prompt() -> str:
    # ≈4k tokens of context -> compute-bound prefill (kept modest to fit small
    # context windows; the point is a big prompt, not the largest possible one)
    return (_PARAGRAPH * 120) + "\nReply with a single word: acknowledged."


def _decode_prompt() -> str:
    # tiny prompt, ask for a long continuation -> memory-bound decode
    return "Write an extremely long, detailed, continuous story. Do not stop early."


def _balanced_prompt() -> str:
    return (_PARAGRAPH * 12) + "\nSummarize the passage in a few sentences."


def text_cells() -> list[LoadCell]:
    """The Core-4 battery's three LOAD cells (idle is measured separately)."""
    return [
        LoadCell("prefill", _prefill_prompt(), max_tokens=8),
        LoadCell("decode", _decode_prompt(), max_tokens=1024),
        LoadCell("balanced", _balanced_prompt(), max_tokens=256),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_battery.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/battery.py tests/test_battery.py
git commit -m "feat(calib): Core-4 text load cells + standardized prompts"
```

---

### Task 2: `LoadClient` — `ChatResult`, protocol, `HttpLoadClient`, `FakeLoadClient` (`battery.py`)

**Files:**
- Modify: `src/tokenwatt/battery.py`
- Test: `tests/test_battery.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ChatResult(tok_in: int | None, tok_out: int | None)` — frozen dataclass; `None` means the upstream reported no usage (never a fabricated `0`).
  - `LoadClient` — `@runtime_checkable` Protocol: `chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult`.
  - `HttpLoadClient(upstream: str, client: httpx.Client | None = None, timeout: float = 120.0)` with `chat(...)` (POST `{upstream}/v1/chat/completions`, `stream=false`, reads `usage`) and `close()`.
  - `FakeLoadClient(tok_in: int = 100, tok_out: int = 50, latency_s: float = 1.0, clock=None)` — returns a fixed `ChatResult`; if `clock` is given, advances it by `latency_s` per `chat()` (so `run_cell`'s wall-clock loop terminates under a fake clock, the way a real HTTP request consumes real time).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_battery.py
import httpx
import pytest

from tokenwatt.battery import ChatResult, LoadClient, HttpLoadClient, FakeLoadClient


def _mock_chat(body, status=200):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = __import__("json").loads(request.content.decode())
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


_CHAT_BODY = {"choices": [{"message": {"content": "acknowledged"}}],
              "usage": {"prompt_tokens": 4123, "completion_tokens": 8}}


def test_http_load_client_posts_chat_and_reads_usage():
    client, seen = _mock_chat(_CHAT_BODY)
    r = HttpLoadClient("http://up", client=client).chat("m1", "hello", max_tokens=8)
    assert r == ChatResult(tok_in=4123, tok_out=8)
    assert seen["url"] == "http://up/v1/chat/completions"
    assert seen["json"]["model"] == "m1"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["stream"] is False
    assert seen["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_http_load_client_raises_on_http_error():
    client, _ = _mock_chat({"error": "boom"}, status=500)
    with pytest.raises(httpx.HTTPStatusError):
        HttpLoadClient("http://up", client=client).chat("m1", "hi", max_tokens=8)


def test_http_load_client_returns_none_when_usage_missing():
    # some local servers omit `usage` on non-streaming responses — never fabricate a 0
    client, _ = _mock_chat({"choices": [{"message": {"content": "ok"}}]})
    assert HttpLoadClient("http://up", client=client).chat("m", "p", 8) == ChatResult(None, None)


def test_http_load_client_tolerates_null_usage():
    # `"usage": null` (key present, value null) must not crash and must read as unknown
    client, _ = _mock_chat({"choices": [], "usage": None})
    assert HttpLoadClient("http://up", client=client).chat("m", "p", 8) == ChatResult(None, None)


def test_http_load_client_satisfies_protocol():
    client, _ = _mock_chat(_CHAT_BODY)
    assert isinstance(HttpLoadClient("http://up", client=client), LoadClient)


def test_fake_load_client_returns_fixed_tokens_and_advances_clock():
    class _Clock:
        def __init__(self): self.t = 0.0
        def monotonic(self): return self.t
        def sleep(self, s): self.t += s

    clk = _Clock()
    fake = FakeLoadClient(tok_in=10, tok_out=20, latency_s=2.0, clock=clk)
    assert fake.chat("m", "p", 8) == ChatResult(tok_in=10, tok_out=20)
    assert clk.t == 2.0          # a request consumed 2s of (fake) wall time
    assert isinstance(fake, LoadClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_battery.py -k "load_client or protocol" -v`
Expected: FAIL — `ImportError: cannot import name 'ChatResult'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tokenwatt/battery.py
from typing import Protocol, runtime_checkable

import httpx


@dataclass(frozen=True)
class ChatResult:
    tok_in: int | None      # None when the upstream reports no usage — never a fabricated 0
    tok_out: int | None


@runtime_checkable
class LoadClient(Protocol):
    def chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult: ...


class HttpLoadClient:
    """Drives inference load by POSTing OpenAI chat completions to an upstream.
    Non-streaming so token usage is read straight from the response body."""

    def __init__(self, upstream: str, client: httpx.Client | None = None, timeout: float = 120.0) -> None:
        self._upstream = upstream.rstrip("/")
        self._timeout = timeout
        self._own = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult:
        r = self._client.post(
            f"{self._upstream}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "stream": False, "temperature": 0.0},
            timeout=self._timeout,
        )
        r.raise_for_status()
        usage = r.json().get("usage")            # None if the key is missing OR explicitly null
        if not usage:                            # None / null / {} -> unknown, NOT a fabricated 0
            return ChatResult(tok_in=None, tok_out=None)
        pin, pout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        return ChatResult(tok_in=int(pin) if pin is not None else None,
                          tok_out=int(pout) if pout is not None else None)

    def close(self) -> None:
        if self._own:
            self._client.close()


class FakeLoadClient:
    """Deterministic test double: fixed token counts, and (given a fake clock)
    advances it by `latency_s` per call so a wall-clock load loop terminates."""

    def __init__(self, tok_in: int = 100, tok_out: int = 50,
                 latency_s: float = 1.0, clock=None) -> None:
        self._r = ChatResult(tok_in=tok_in, tok_out=tok_out)
        self._latency = latency_s
        self._clock = clock

    def chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult:
        if self._clock is not None:
            self._clock.sleep(self._latency)
        return self._r
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_battery.py -v`
Expected: PASS (9 passed in the file).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/battery.py tests/test_battery.py
git commit -m "feat(calib): LoadClient (HTTP chat driver + fake)"
```

---

### Task 3: `IdleRates`, `CellSample`, `measure_idle`, `run_cell` (`campaign.py`)

**Files:**
- Modify: `src/tokenwatt/campaign.py`
- Test: `tests/test_campaign_run.py`

**Interfaces:**
- Consumes: `EnergyByRail`, `EnergyMeter` (`meter.py`); `MeterSource` (`metersource.py`); `ChatResult` (`battery.py`).
- Produces:
  - `IdleRates(rail_w: dict[str, float], wall_w: float, dt_s: float)`.
  - `CellSample(cell: str, model: str, dt_s: float, e_rail_marginal_j: dict[str, float], e_wall_marginal_j: float, tok_in: int | None, tok_out: int | None, requests: int)` — `requests` is the count of load calls (always known, usage-independent); `tok_*` is `None` if any request lacked usage.
  - `measure_idle(meter, source, *, seconds: float = 300.0, sleep=time.sleep, monotonic=time.monotonic) -> IdleRates` — bracket an idle window, return per-rail idle watts + idle wall watts.
  - `run_cell(meter, source, load_fn, idle: IdleRates, *, cell: str, model: str, seconds: float = 300.0, sleep=time.sleep, monotonic=time.monotonic) -> CellSample` — where `load_fn: Callable[[], ChatResult]` is called in a loop for `seconds`; returns the marginal sample.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_campaign_run.py
import pytest

from tokenwatt.meter import FakeMeter, EnergyByRail
from tokenwatt.metersource import FakeMeterSource
from tokenwatt.battery import ChatResult
from tokenwatt import campaign


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_measure_idle_returns_per_rail_and_wall_watts():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))  # +10 J/read
    source = FakeMeterSource([100.0, 100.05])                            # +0.05 Wh over window
    idle = campaign.measure_idle(meter, source, seconds=10.0,
                                 sleep=clk.sleep, monotonic=clk.monotonic)
    assert idle.dt_s == pytest.approx(10.0)
    assert idle.rail_w == {"cpu_total": pytest.approx(1.0)}      # 10 J / 10 s
    assert idle.wall_w == pytest.approx(0.05 * 3600 / 10.0)      # 18 W


def test_run_cell_brackets_load_and_subtracts_idle_from_both_sides():
    clk = _Clock()
    # rail rises 10 J between the two cumulative() reads (start, end)
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.000, 100.100])                # +0.1 Wh -> 360 J wall
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.5}, wall_w=3.0, dt_s=10.0)
    # each load call advances the clock 2 s and returns 7 in / 11 out tokens
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=7, tok_out=11)

    s = campaign.run_cell(meter, source, load_fn, idle, cell="prefill", model="m1",
                          seconds=10.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.cell == "prefill" and s.model == "m1"
    assert s.dt_s == pytest.approx(10.0)
    # 5 load calls fit in 10 s at 2 s each
    assert (s.tok_in, s.tok_out) == (35, 55)
    assert s.requests == 5                          # always-known load-validity signal
    # marginal rail = ΔE_rail − idle_rail·dt = 10 − 0.5·10 = 5 J
    assert s.e_rail_marginal_j == {"cpu_total": pytest.approx(5.0)}
    # marginal wall = ΔWh·3600 − idle_wall·dt = 360 − 3·10 = 330 J
    assert s.e_wall_marginal_j == pytest.approx(330.0)


def test_run_cell_records_tokens_unknown_but_still_counts_requests():
    # an upstream that withholds usage -> tok_in/out None (never a fake 0), but the
    # request count still proves sustained load ran (the efficacy signal)
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([1.0, 1.1])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=None, tok_out=None)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="decode", model="m",
                          seconds=6.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.tok_in is None and s.tok_out is None    # honest unknown, not 0
    assert s.requests == 3                            # load still ran and was counted
    assert s.e_wall_marginal_j >= 0.0                # energy sample still valid


def test_run_cell_clamps_negative_marginals_to_zero():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 1.0}))   # tiny rail move
    source = FakeMeterSource([50.0, 50.0])                                # wall flat
    idle = campaign.IdleRates(rail_w={"cpu_total": 100.0}, wall_w=100.0, dt_s=1.0)
    def load_fn():
        clk.sleep(1.0)
        return ChatResult(tok_in=1, tok_out=1)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="decode", model="m",
                          seconds=1.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.e_rail_marginal_j["cpu_total"] == 0.0     # never a negative energy
    assert s.e_wall_marginal_j == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign_run.py -v`
Expected: FAIL — `AttributeError: module 'tokenwatt.campaign' has no attribute 'measure_idle'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tokenwatt/campaign.py (time, dataclass already imported at top)
from typing import Callable

from tokenwatt.battery import ChatResult


@dataclass
class IdleRates:
    rail_w: dict[str, float]     # per-rail idle power (W)
    wall_w: float                # idle wall power (W)
    dt_s: float


@dataclass
class CellSample:
    cell: str
    model: str
    dt_s: float
    e_rail_marginal_j: dict[str, float]   # per-rail ΔE − idle_rail·dt, clamped ≥ 0
    e_wall_marginal_j: float              # ΔWh·3600 − idle_wall·dt, clamped ≥ 0
    tok_in: int | None                    # None if any request in the cell lacked usage
    tok_out: int | None
    requests: int                         # count of load calls — proof sustained load ran


def measure_idle(meter: EnergyMeter, source: MeterSource, *, seconds: float = 300.0,
                 sleep=time.sleep, monotonic=time.monotonic) -> IdleRates:
    """Bracket an idle window (no inference) → per-rail idle watts + idle wall watts.
    Defines the marginal baseline subtracted from every load cell."""
    t0 = monotonic()
    e0 = meter.cumulative()
    w0 = source.read_accumulated_wh()
    sleep(seconds)
    dt = max(monotonic() - t0, 1e-9)
    e_rail = meter.cumulative() - e0
    wall_j = max(source.read_accumulated_wh() - w0, 0.0) * 3600.0
    return IdleRates(rail_w={r: j / dt for r, j in e_rail.joules.items()},
                     wall_w=wall_j / dt, dt_s=dt)


def run_cell(meter: EnergyMeter, source: MeterSource, load_fn: Callable[[], ChatResult],
             idle: IdleRates, *, cell: str, model: str, seconds: float = 300.0,
             sleep=time.sleep, monotonic=time.monotonic) -> CellSample:
    """Drive `load_fn` in a loop for `seconds` while bracketing rail+wall energy,
    then return the idle-subtracted marginal sample. `load_fn` is expected to
    consume wall-clock time (a real HTTP request does; the fake advances the clock)."""
    t0 = monotonic()
    e0 = meter.cumulative()
    w0 = source.read_accumulated_wh()
    tok_in = tok_out = 0
    requests = 0
    any_unknown = False
    while monotonic() - t0 < seconds:
        r = load_fn()
        requests += 1
        if r.tok_in is None or r.tok_out is None:
            any_unknown = True                 # honest: unknown, not a fabricated 0
        else:
            tok_in += r.tok_in
            tok_out += r.tok_out
    dt = max(monotonic() - t0, 1e-9)
    e_rail = meter.cumulative() - e0
    wall_j = max(source.read_accumulated_wh() - w0, 0.0) * 3600.0
    rails = set(e_rail.joules) | set(idle.rail_w)
    marg_rail = {r: max(e_rail.joules.get(r, 0.0) - idle.rail_w.get(r, 0.0) * dt, 0.0)
                 for r in rails}
    marg_wall = max(wall_j - idle.wall_w * dt, 0.0)
    return CellSample(cell=cell, model=model, dt_s=dt, e_rail_marginal_j=marg_rail,
                      e_wall_marginal_j=marg_wall,
                      tok_in=None if any_unknown else tok_in,
                      tok_out=None if any_unknown else tok_out, requests=requests)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign_run.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/campaign.py tests/test_campaign_run.py
git commit -m "feat(calib): idle baseline + per-cell marginal bracketing"
```

---

### Task 4: `CampaignResult`, `run_campaign`, persistence (`campaign.py`)

**Files:**
- Modify: `src/tokenwatt/campaign.py`
- Test: `tests/test_campaign_run.py`

**Interfaces:**
- Consumes: `IdleRates`, `CellSample`, `measure_idle`, `run_cell` (Task 3); `LoadCell`, `LoadClient` (`battery.py`).
- Produces:
  - `CampaignResult(model, meter_name, meter_tier, meter_accuracy_pct, cell_seconds, passes, idle: IdleRates, samples: list[CellSample], schema_version: int, timestamp: float)`.
  - `run_campaign(*, cells: list[LoadCell], model: str, load: LoadClient, make_meter=_default_meter, make_source=_default_shelly, host, switch_id=0, password=None, cell_seconds=300.0, passes=2, timestamp, idle_seconds=None, sleep=time.sleep, monotonic=time.monotonic) -> tuple[CampaignResult | None, str]` — `make_meter`/`make_source` default to the same private constructors `run_probe` uses, so the CLI calls it without them (a true mirror of `run_probe`). — preflight (reachable + meter), measure idle, run each `(cell × pass)`, collect samples; `(None, reason)` on preflight failure or a mid-window read error.
  - `campaign_to_dict(result: CampaignResult) -> dict` and `write_campaign(result: CampaignResult, path: str) -> None` (JSON, dirs auto-created).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_campaign_run.py
import json
import os

from tokenwatt.battery import FakeLoadClient, LoadCell


def _cells():
    return [LoadCell("prefill", "p", 8), LoadCell("decode", "d", 1024)]


def test_run_campaign_collects_one_sample_per_cell_per_pass():
    clk = _Clock()
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        host="h", cell_seconds=4.0, passes=2, timestamp=1_700_000_000.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert len(result.samples) == 4                       # 2 cells × 2 passes
    assert [s.cell for s in result.samples] == ["prefill", "decode", "prefill", "decode"]
    assert result.model == "m1" and result.passes == 2 and result.cell_seconds == 4.0
    assert result.meter_tier == "fake"                    # from the FakeMeterSource
    assert all(s.tok_in > 0 for s in result.samples)      # load actually ran


def test_run_campaign_aborts_loud_when_meter_source_unreachable():
    def _dead_source(*_a, **_k):
        s = FakeMeterSource([1.0])
        s.reachable = lambda: (False, "ConnectError")   # type: ignore[attr-defined]
        return s
    result, msg = campaign.run_campaign(
        cells=_cells(), model="m", load=FakeLoadClient(),
        make_meter=lambda: FakeMeter(), make_source=_dead_source,
        host="h", timestamp=0.0)
    assert result is None and "unreachable" in msg.lower()


def test_write_campaign_round_trips_json(tmp_path):
    clk = _Clock()
    result, _ = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        host="h", cell_seconds=4.0, passes=1, timestamp=1_700_000_000.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    out = os.path.join(tmp_path, "sub", "campaign.json")
    campaign.write_campaign(result, out)
    doc = json.load(open(out))
    assert doc["model"] == "m1"
    assert doc["meter"]["tier"] == "fake"
    assert len(doc["samples"]) == 2
    assert doc["samples"][0]["cell"] == "prefill"
    assert "e_wall_marginal_j" in doc["samples"][0]
    assert "requests" in doc["samples"][0]        # efficacy signal persisted for C2
    assert "wall_w" in doc["idle"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign_run.py -k "campaign" -v`
Expected: FAIL — `AttributeError: module 'tokenwatt.campaign' has no attribute 'run_campaign'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tokenwatt/campaign.py (add `import json`, `import os` at top with the others)
from dataclasses import asdict

from tokenwatt.battery import LoadCell, LoadClient

_CAMPAIGN_SCHEMA = 1


@dataclass
class CampaignResult:
    model: str
    meter_name: str
    meter_tier: str
    meter_accuracy_pct: float
    cell_seconds: float
    passes: int
    idle: IdleRates
    samples: list[CellSample]
    schema_version: int
    timestamp: float


def run_campaign(*, cells: list[LoadCell], model: str, load: LoadClient,
                 make_meter=_default_meter, make_source=_default_shelly, host: str,
                 switch_id: int = 0, password: str | None = None,
                 cell_seconds: float = 300.0, passes: int = 2,
                 timestamp: float, idle_seconds: float | None = None,
                 sleep=time.sleep, monotonic=time.monotonic) -> tuple["CampaignResult | None", str]:
    """Preflight, measure idle, then run each (cell × pass) as a sustained-load
    bracket. Returns (result, message); (None, reason) on preflight or read failure."""
    source = make_source(host, switch_id, password)
    ok, detail = source.reachable() if hasattr(source, "reachable") else (True, "")
    if not ok:
        return None, f"meter unreachable: {detail}"
    try:
        meter = make_meter()
    except Exception as e:
        return None, (f"energy meter unavailable ({type(e).__name__}: {e}); "
                      f"run this on the Apple-Silicon Mac being calibrated")
    try:
        idle = measure_idle(meter, source, seconds=idle_seconds or cell_seconds,
                            sleep=sleep, monotonic=monotonic)
        samples: list[CellSample] = []
        for _ in range(passes):
            for cell in cells:
                samples.append(run_cell(
                    meter, source, lambda c=cell: load.chat(model, c.prompt, c.max_tokens),
                    idle, cell=cell.name, model=model, seconds=cell_seconds,
                    sleep=sleep, monotonic=monotonic))
    except Exception as e:
        return None, f"campaign read failed ({type(e).__name__}: {e})"
    result = CampaignResult(
        model=model, meter_name=source.name, meter_tier=source.tier,
        meter_accuracy_pct=source.accuracy_pct, cell_seconds=cell_seconds, passes=passes,
        idle=idle, samples=samples, schema_version=_CAMPAIGN_SCHEMA, timestamp=timestamp)
    return result, f"{len(samples)} samples over {passes} pass(es)"


def campaign_to_dict(result: CampaignResult) -> dict:
    return {
        "schema_version": result.schema_version,
        "timestamp": result.timestamp,
        "model": result.model,
        "meter": {"name": result.meter_name, "tier": result.meter_tier,
                  "accuracy_pct": result.meter_accuracy_pct},
        "cell_seconds": result.cell_seconds,
        "passes": result.passes,
        "idle": asdict(result.idle),
        "samples": [asdict(s) for s in result.samples],
    }


def write_campaign(result: CampaignResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(campaign_to_dict(result), f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign_run.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/campaign.py tests/test_campaign_run.py
git commit -m "feat(calib): run_campaign + JSON sample persistence"
```

---

### Task 5: `calibrate campaign` CLI (`cli.py`)

**Files:**
- Modify: `src/tokenwatt/cli.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `run_campaign`, `write_campaign`, `CampaignResult` (`campaign.py`); `text_cells`, `HttpLoadClient` (`battery.py`); `load_config`, `ConfigError` (`config.py`); the existing `calibrate_app` typer sub-app (C0).
- Produces: CLI `tokenwatt calibrate campaign --meter-host H --upstream URL --model M [--meter-id N] [--cell-seconds 300] [--passes 2] [--out PATH] [--config PATH]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli_smoke.py
def test_calibrate_campaign_help_lists_upstream_and_model():
    res = runner.invoke(app, ["calibrate", "campaign", "--help"])
    assert res.exit_code == 0
    assert "--upstream" in res.output and "--model" in res.output


def test_calibrate_campaign_requires_upstream_and_model():
    res = runner.invoke(app, ["calibrate", "campaign", "--meter-host", "h"])
    assert res.exit_code == 1
    assert "upstream" in res.output.lower() or "model" in res.output.lower()
```

*(`runner = CliRunner()` and `from tokenwatt.cli import app` already exist at the top of this file from C0.)*

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py -k campaign -v`
Expected: FAIL — no such command `campaign` (usage error / exit 2).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tokenwatt/cli.py, after the existing `calibrate probe` command
@calibrate_app.command("campaign")
def calibrate_campaign(
    meter_host: Optional[str] = typer.Option(None, "--meter-host", help="Shelly smart plug host/IP"),
    meter_id: int = typer.Option(0, "--meter-id", help="RPC Switch component id"),
    upstream: Optional[str] = typer.Option(None, "--upstream", help="inference server base URL (http://host:port)"),
    model: Optional[str] = typer.Option(None, "--model", help="model id to send in requests"),
    cell_seconds: float = typer.Option(300.0, "--cell-seconds", help="sustained-load seconds per cell (keep >> plug cadence)"),
    passes: int = typer.Option(2, "--passes", help="battery repeats (repeatability)"),
    out: Optional[str] = typer.Option(None, "--out", help="where to write the samples JSON"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="read meter host from this config"),
):
    """Run the Core-4 text load battery, capture idle-subtracted marginal rail/wall
    samples, and write them for the C2 fit. RAW sample collection — not a calibration."""
    import os
    import time as _time
    from tokenwatt import campaign
    from tokenwatt.battery import text_cells, HttpLoadClient
    from tokenwatt.config import load_config, ConfigError

    host, sid, password = meter_host, meter_id, None
    if host is None and config is not None:
        try:
            cfg = load_config(config)
        except ConfigError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        host, sid, password = cfg.calibration.meter_host, cfg.calibration.meter_id, cfg.calibration.meter_password
    if not host:
        typer.echo("no meter host — pass --meter-host or set calibration.meter_host in your config", err=True)
        raise typer.Exit(1)
    if not upstream or not model:
        typer.echo("both --upstream and --model are required (the load battery hits that model on that server)", err=True)
        raise typer.Exit(1)

    ts = _time.time()
    out_path = out or os.path.expanduser(f"~/.tokenwatt/calibration/campaign-{int(ts)}.json")
    load = HttpLoadClient(upstream)
    try:
        result, msg = campaign.run_campaign(
            cells=text_cells(), model=model, load=load, host=host, switch_id=sid,
            password=password, cell_seconds=cell_seconds, passes=passes, timestamp=ts)
    finally:
        load.close()
    typer.echo(msg, err=(result is None))
    if result is None:
        raise typer.Exit(1)
    campaign.write_campaign(result, out_path)
    typer.echo(f"wrote {out_path}")
    typer.echo(f"idle: wall {result.idle.wall_w:.2f} W, rails {result.idle.rail_w}")
    for s in result.samples:
        tok = "unknown" if s.tok_in is None else f"{s.tok_in}/{s.tok_out}"   # never a fake 0
        typer.echo(f"  {s.cell:<9} marg wall {s.e_wall_marginal_j:8.1f} J   "
                   f"rail {sum(s.e_rail_marginal_j.values()):7.1f} J   "
                   f"req {s.requests:>3}   tok {tok}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_smoke.py -k campaign -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new ones; no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/cli.py tests/test_cli_smoke.py
git commit -m "feat(calib): tokenwatt calibrate campaign (Core-4 sample collection)"
```

---

## On-device acceptance (run on the M3 Ultra, with the Shelly + an inference server)

Not CI — the operator run that closes C1 and produces the first real sample set for C2.

1. Bring up a text inference server (e.g. `mlx-openai-server`) and note its base URL + model id.
2. Quiesce the machine (close heavy apps) so the idle baseline is clean.
3. Run a **short shake-out** first to prove the pipeline end-to-end without a 40-min wait:
   `tokenwatt calibrate campaign --meter-host <plug-ip> --upstream <url> --model <id> --cell-seconds 90 --passes 1`
   Confirm: idle wall W is plausible; each cell's **marginal wall J > 0** and rises with load; `decode`/`prefill` marginals differ (rails moved differently); a JSON file is written.
4. Run the real **Core-4** campaign: `--cell-seconds 300 --passes 2` (~40 min). Leave the machine otherwise idle.
5. Sanity-check the two passes agree roughly per cell (repeatability). Keep the JSON — it is C2's fit input.
6. Note whether `decode` (memory-bound) vs `prefill` (compute-bound) produce *distinguishable* marginal rail signatures — early read on whether C5's per-rail tier will be identifiable.

---

## Self-Review

**Spec coverage (§6 Component B):**
- Load cells `prefill`/`decode`/`balanced` with the specified prompt/token shapes → Task 1; `idle` measured separately → Task 3 `measure_idle`. `embeddings`/`vision` explicitly deferred (Core-4 per the user; documented).
- Standardized, deterministic, in-package prompts; model is the variable → Task 1 + the `--model`/`--upstream` CLI.
- Sustained load loop for a fixed `Δt` → Task 3 `run_cell`.
- Sample shape `{cell, model, dt, e_rail_marginal{...}, e_wall_marginal, tok_in, tok_out}` → Task 3 `CellSample`, refined per the honesty contract: `tok_*` are `int | None` (unknown, never a fake 0, when a server omits `usage`) and a `requests` count is added (usage-independent proof sustained load ran). Tokens are validity/bookkeeping — the fit uses energy; runtime $/token is computed separately by the proxy from real traffic.
- Idle measured at **both** rails and wall; marginal subtraction on both sides → Task 3 `measure_idle` + `run_cell`; energy Wh→J ×3600; clamp ≥0.
- Persisted for the C2 fit → Task 4 `write_campaign` (single JSON per campaign, with meter/idle/samples + `schema_version`).
- Honesty: no fit, no band, no `calibrated` label anywhere → constraint enforced; CLI prints raw marginals only.

**Placeholder scan:** none — every step has runnable code + an exact command with expected output.

**Type consistency:** `ChatResult(tok_in, tok_out)`, `LoadClient.chat(model, prompt, max_tokens) -> ChatResult`, `IdleRates(rail_w, wall_w, dt_s)`, `CellSample(cell, model, dt_s, e_rail_marginal_j, e_wall_marginal_j, tok_in, tok_out)`, `run_cell(meter, source, load_fn, idle, *, cell, model, seconds, ...)`, and `run_campaign(*, cells, model, load, make_meter, make_source, host, ...)` are used identically across Tasks 1→5; `run_campaign` mirrors C0's `run_probe` `(result|None, message)` seam; `make_source(host, switch_id, password)` matches C0's `_default_shelly` signature.

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, spec+quality review between tasks, Opus whole-branch review at the end.
2. **Inline Execution** — tasks in this session via executing-plans, batched with checkpoints.
