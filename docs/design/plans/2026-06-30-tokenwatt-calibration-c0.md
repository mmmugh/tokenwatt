# TokenWatt Calibration C0 — Shelly Link + Raw Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the wall-energy `MeterSource` abstraction and a `ShellyMeterSource` over Gen2 RPC, then ship `tokenwatt calibrate probe` — a diagnostic that reads synchronized rail-Δ (zeus) and wall-Δ (plug) over a window and reports whether they move together, plus the plug's measured resolution/cadence.

**Architecture:** A new `metersource.py` holds the `MeterSource` protocol + `ShellyMeterSource` + a deterministic `FakeMeterSource` (mirroring how `FakeMeter` lives beside `ZeusMeter` in `meter.py`). A new `campaign.py` holds the harness orchestration — a pure, dependency-injected `probe()` that polls both meters on one clock — and grows into the C1 battery later. The CLI gets a `calibrate` typer sub-app whose first command is `probe`. Nothing touches the proxy hot path; this is offline tooling.

**Tech Stack:** Python ≥3.10, `httpx` (sync `Client`, already a dependency), `typer`, `pydantic`, `pytest` + `pytest-asyncio`. Shelly Plus Plug US Gen2 RPC (`GET /rpc/Switch.GetStatus?id=<n>` → unwrapped status object with `aenergy.total` in Wh and `apower` in W).

## Global Constraints

- **Python floor:** `requires-python>=3.10`; all code must run on 3.10–3.14 (CI matrix).
- **Apple-Silicon meter is gated:** `ZeusMeter` is only constructed on the Mac being calibrated; off-platform it raises on construction and the harness must degrade with a clear message — never crash. Tests use fakes and run anywhere.
- **No sudo** anywhere (consistent with the project's metering ethos).
- **Honesty contract:** the probe reports *raw* measured numbers and a crude `wall/rail` ratio only — it is **not** a calibration and must never emit a `calibrated` label, a band, or a fitted coefficient. That is C2+.
- **Read the plug's own counter:** integrate energy from `aenergy.total`, never by integrating instantaneous `apower` ourselves.
- **Deterministic CI:** the whole probe path runs under `FakeMeter` + `FakeMeterSource` + an injected clock; no network, no hardware, no wall-clock sleeps in tests.
- **`VERSION` auto-bumps** its patch on every commit via `.githooks/pre-commit` — expected; let it run.
- **Match existing style:** `from __future__ import annotations`; `@runtime_checkable` protocols like `EnergyMeter`; dependency-injected seams like `doctor.run(...)`; tiny inline test fakes like `tests/test_doctor.py`.

---

## File Structure

- **Create `src/tokenwatt/metersource.py`** — `MeterSource` protocol; `ShellyStatus`; `ShellyMeterSource` (Gen2 RPC client + `reachable()`); `FakeMeterSource`. Single responsibility: *the wall-energy reference*.
- **Create `src/tokenwatt/campaign.py`** — `ProbeResult`; pure `probe()`; `run_probe()` construction seam; `format_probe()`. Single responsibility: *harness orchestration* (probe now; battery in C1).
- **Modify `src/tokenwatt/config.py`** — add `CalibrationConfig` + `Config.calibration` (just the meter connection for now).
- **Modify `src/tokenwatt/cli.py`** — add the `calibrate` typer sub-app + `probe` command.
- **Create `tests/test_metersource.py`**, **`tests/test_campaign.py`**; **extend `tests/test_config.py`** and add a probe smoke check.

**Deferred out of C0 (do not build here):** `ManualMeterSource` and the `--meter manual` fallback (C4), `machine_id` detection (C2), the fit / `calibration.py` (C2), the profile registry (C2), runtime `Cal` + `calib_*` ledger columns (C3), the `calibrate run` wizard and `calibrate show` (C4), per-rail gating (C5).

---

### Task 1: `MeterSource` protocol + `FakeMeterSource`

**Files:**
- Create: `src/tokenwatt/metersource.py`
- Test: `tests/test_metersource.py`

**Interfaces:**
- Produces:
  - `MeterSource` — `@runtime_checkable` Protocol with attributes `name: str`, `accuracy_pct: float`, `tier: str` and method `read_accumulated_wh() -> float`.
  - `FakeMeterSource(readings: list[float])` — deterministic; `read_accumulated_wh()` returns successive readings, holding the last when exhausted.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metersource.py
import pytest

from tokenwatt.metersource import MeterSource, FakeMeterSource


def test_fake_meter_source_returns_scripted_readings_in_order():
    src = FakeMeterSource([100.0, 100.001, 100.5])
    assert src.read_accumulated_wh() == 100.0
    assert src.read_accumulated_wh() == 100.001
    assert src.read_accumulated_wh() == 100.5


def test_fake_meter_source_holds_last_reading_when_exhausted():
    src = FakeMeterSource([7.0])
    assert src.read_accumulated_wh() == 7.0
    assert src.read_accumulated_wh() == 7.0  # does not raise; sustained-reads safe


def test_fake_meter_source_satisfies_protocol():
    # the protocol is the contract the harness depends on; a source missing
    # name/tier/accuracy or read_accumulated_wh must NOT type as a MeterSource
    assert isinstance(FakeMeterSource([1.0]), MeterSource)


def test_fake_meter_source_rejects_empty_readings():
    with pytest.raises(ValueError):
        FakeMeterSource([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metersource.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.metersource'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/metersource.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MeterSource(Protocol):
    """A wall-energy reference. We integrate energy by reading the meter's own
    monotonic accumulated-watt-hour counter at window boundaries — never by
    integrating instantaneous power ourselves."""
    name: str
    accuracy_pct: float          # the meter's own ± accuracy; becomes the band floor in C2
    tier: str                    # "smart_plug" | "manual" | "lab" | "fake"

    def read_accumulated_wh(self) -> float: ...


class FakeMeterSource:
    """Deterministic test double: replays a scripted Wh accumulator, holding the
    last value once exhausted so sustained polling never raises."""
    name = "fake"
    accuracy_pct = 0.0
    tier = "fake"

    def __init__(self, readings: list[float]) -> None:
        if not readings:
            raise ValueError("FakeMeterSource needs at least one reading")
        self._readings = list(readings)
        self._i = 0

    def read_accumulated_wh(self) -> float:
        v = self._readings[min(self._i, len(self._readings) - 1)]
        self._i += 1
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metersource.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/metersource.py tests/test_metersource.py
git commit -m "feat(calib): MeterSource protocol + FakeMeterSource"
```

---

### Task 2: `ShellyMeterSource` (Gen2 RPC client)

**Files:**
- Modify: `src/tokenwatt/metersource.py`
- Test: `tests/test_metersource.py`

**Interfaces:**
- Consumes: `MeterSource` (Task 1).
- Produces:
  - `ShellyStatus(total_wh: float, apower_w: float, voltage_v: float)` — frozen dataclass.
  - `ShellyMeterSource(host: str, switch_id: int = 0, password: str | None = None, client: httpx.Client | None = None, timeout: float = 2.0)` with `read_status() -> ShellyStatus`, `read_accumulated_wh() -> float`, `reachable() -> tuple[bool, str]`, `close() -> None`. Class attrs `name="shelly-plus-plug-us"`, `accuracy_pct=1.0`, `tier="smart_plug"`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_metersource.py
import httpx

from tokenwatt.metersource import ShellyMeterSource, ShellyStatus, MeterSource


_SHELLY_BODY = {       # shape of GET /rpc/Switch.GetStatus?id=0 (Gen2, unwrapped)
    "id": 0, "output": True, "apower": 12.4, "voltage": 121.7, "current": 0.10,
    "aenergy": {"total": 1234.567, "by_minute": [0, 0, 0], "minute_ts": 1700000000},
}


def _mock_shelly(body=_SHELLY_BODY, status=200):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, seen


def test_shelly_reads_accumulated_wh_from_aenergy_total():
    client, seen = _mock_shelly()
    src = ShellyMeterSource("shelly.local", switch_id=0, client=client)
    assert src.read_accumulated_wh() == pytest.approx(1234.567)
    # hits the Gen2 RPC GET endpoint with the right component id
    assert seen["url"] == "http://shelly.local/rpc/Switch.GetStatus?id=0"


def test_shelly_read_status_parses_power_and_voltage():
    client, _ = _mock_shelly()
    src = ShellyMeterSource("shelly.local", client=client)
    s = src.read_status()
    assert isinstance(s, ShellyStatus)
    assert (s.total_wh, s.apower_w, s.voltage_v) == pytest.approx((1234.567, 12.4, 121.7))


def test_shelly_satisfies_meter_source_protocol():
    client, _ = _mock_shelly()
    assert isinstance(ShellyMeterSource("h", client=client), MeterSource)


def test_shelly_reachable_true_summarizes_status():
    client, _ = _mock_shelly()
    ok, detail = ShellyMeterSource("h", client=client).reachable()
    assert ok is True
    assert "1234.567" in detail and "12.4" in detail


def test_shelly_reachable_false_on_http_error():
    client, _ = _mock_shelly(status=500)
    ok, detail = ShellyMeterSource("h", client=client).reachable()
    assert ok is False
    assert "500" in detail or "Error" in detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metersource.py -v`
Expected: FAIL — `ImportError: cannot import name 'ShellyMeterSource'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tokenwatt/metersource.py
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ShellyStatus:
    total_wh: float       # aenergy.total — the monotonic accumulator we integrate from
    apower_w: float       # instantaneous active power (sanity / live display only)
    voltage_v: float


class ShellyMeterSource:
    """Shelly Plus Plug US (Gen2) wall-energy meter over local RPC. Reads the
    plug's own accumulated-energy counter; no cloud, no sudo."""
    name = "shelly-plus-plug-us"
    accuracy_pct = 1.0                       # datasheet ~±1%; band floor in C2
    tier = "smart_plug"

    def __init__(self, host: str, switch_id: int = 0, password: str | None = None,
                 client: httpx.Client | None = None, timeout: float = 2.0) -> None:
        self._host = host.rstrip("/")
        self._id = switch_id
        self._timeout = timeout
        self._own = client is None
        auth = httpx.DigestAuth("admin", password) if password else None
        self._client = client if client is not None else httpx.Client(auth=auth, timeout=timeout)

    def _url(self) -> str:
        base = self._host if "://" in self._host else f"http://{self._host}"
        return f"{base}/rpc/Switch.GetStatus?id={self._id}"

    def read_status(self) -> ShellyStatus:
        r = self._client.get(self._url(), timeout=self._timeout)
        r.raise_for_status()
        body = r.json()
        return ShellyStatus(
            total_wh=float(body["aenergy"]["total"]),
            apower_w=float(body.get("apower", 0.0)),
            voltage_v=float(body.get("voltage", 0.0)),
        )

    def read_accumulated_wh(self) -> float:
        return self.read_status().total_wh

    def reachable(self) -> tuple[bool, str]:
        try:
            s = self.read_status()
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        return True, f"{self.name} @ {self._host}: {s.total_wh:.3f} Wh total, {s.apower_w:.1f} W now"

    def close(self) -> None:
        if self._own:
            self._client.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metersource.py -v`
Expected: PASS (9 passed total in the file).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/metersource.py tests/test_metersource.py
git commit -m "feat(calib): ShellyMeterSource over Gen2 RPC (aenergy.total)"
```

---

### Task 3: `CalibrationConfig` in config

**Files:**
- Modify: `src/tokenwatt/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `CalibrationConfig(meter_host: str | None = None, meter_id: int = 0, meter_password: str | None = None)` and `Config.calibration: CalibrationConfig` (default-constructed). *(C3 extends this block with profile-loading fields; do not add them now.)*

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_config.py
from tokenwatt.config import Config, CalibrationConfig


def test_calibration_defaults_are_empty():
    # zero-config must still construct; an absent plug is not an error
    cfg = Config()
    assert isinstance(cfg.calibration, CalibrationConfig)
    assert cfg.calibration.meter_host is None
    assert cfg.calibration.meter_id == 0
    assert cfg.calibration.meter_password is None


def test_calibration_loads_from_mapping():
    cfg = Config(calibration={"meter_host": "shelly.local", "meter_id": 1})
    assert cfg.calibration.meter_host == "shelly.local"
    assert cfg.calibration.meter_id == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k calibration -v`
Expected: FAIL — `ImportError: cannot import name 'CalibrationConfig'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/tokenwatt/config.py, after DiscoveryConfig
class CalibrationConfig(BaseModel):
    """Wall-meter connection for `tokenwatt calibrate`. Optional — absent means
    'no plug configured', not an error. (C3 will add profile-loading fields.)"""
    meter_host: str | None = None        # Shelly Plus Plug US host/IP
    meter_id: int = 0                    # Gen2 Switch component id
    meter_password: str | None = None    # set only if the plug has auth enabled
```

```python
# in class Config, add the field (after `discovery: DiscoveryConfig = ...`)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -k calibration -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/config.py tests/test_config.py
git commit -m "feat(calib): calibration config block (meter connection)"
```

---

### Task 4: `probe()` + `ProbeResult` + `format_probe()`

**Files:**
- Create: `src/tokenwatt/campaign.py`
- Test: `tests/test_campaign.py`

**Interfaces:**
- Consumes: `EnergyByRail`, `EnergyMeter`, `FakeMeter` from `tokenwatt.meter`; `MeterSource`, `FakeMeterSource` from `tokenwatt.metersource`.
- Produces:
  - `ProbeResult` dataclass: `dt_s, wall_wh, wall_w, rail_total_j, rail_w, rail_by_rail_j: dict[str,float], ratio_wall_over_rail: float | None, meter_resolution_wh: float | None, meter_cadence_s: float | None, n_samples: int`.
  - `probe(meter, source, *, seconds=30.0, poll_s=0.5, sleep=time.sleep, monotonic=time.monotonic) -> ProbeResult`.
  - `format_probe(r: ProbeResult) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_campaign.py
import pytest

from tokenwatt.meter import FakeMeter, EnergyByRail
from tokenwatt.metersource import FakeMeterSource
from tokenwatt import campaign


class _Clock:
    """Deterministic clock: sleep() advances virtual time; monotonic() reads it."""
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_probe_computes_synchronized_deltas_and_meter_characterization():
    clk = _Clock()
    # one rail rises 10 J per cumulative() read; 5 reads over the window -> 40 J delta
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    # accumulator (Wh) ticks by 0.001 Wh every other poll -> resolution 0.001, cadence 1.0s
    source = FakeMeterSource([100.000, 100.000, 100.001, 100.001, 100.002])

    r = campaign.probe(meter, source, seconds=2.0, poll_s=0.5,
                       sleep=clk.sleep, monotonic=clk.monotonic)

    assert r.n_samples == 5
    assert r.dt_s == pytest.approx(2.0)
    assert r.wall_wh == pytest.approx(0.002, abs=1e-9)
    assert r.wall_w == pytest.approx(0.002 * 3600 / 2.0)        # J/s over the window
    assert r.rail_total_j == pytest.approx(40.0)
    assert r.rail_by_rail_j == {"cpu_total": pytest.approx(40.0)}
    assert r.rail_w == pytest.approx(20.0)
    assert r.ratio_wall_over_rail == pytest.approx(0.002 * 3600 / 40.0)
    # the C0 raison d'être: characterize the plug before trusting it
    assert r.meter_resolution_wh == pytest.approx(0.001, abs=1e-9)
    assert r.meter_cadence_s == pytest.approx(1.0)


def test_probe_ratio_is_none_when_no_rail_movement():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({}))           # rails flat
    source = FakeMeterSource([5.0, 5.0, 5.0])
    r = campaign.probe(meter, source, seconds=1.0, poll_s=0.5,
                       sleep=clk.sleep, monotonic=clk.monotonic)
    assert r.rail_total_j == 0.0
    assert r.ratio_wall_over_rail is None     # never divide-by-zero into a fake ratio


def test_format_probe_is_plain_and_marks_uncalibrated():
    r = campaign.ProbeResult(
        dt_s=2.0, wall_wh=0.002, wall_w=3.6, rail_total_j=40.0, rail_w=20.0,
        rail_by_rail_j={"cpu_total": 40.0}, ratio_wall_over_rail=0.18,
        meter_resolution_wh=0.001, meter_cadence_s=1.0, n_samples=5)
    out = campaign.format_probe(r)
    assert "wall" in out.lower() and "rail" in out.lower()
    assert "cpu_total" in out
    # must not masquerade as a calibration
    assert "calibrated" not in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.campaign'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/campaign.py
from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median

from tokenwatt.meter import EnergyByRail, EnergyMeter
from tokenwatt.metersource import MeterSource


@dataclass
class ProbeResult:
    dt_s: float
    wall_wh: float
    wall_w: float
    rail_total_j: float
    rail_w: float
    rail_by_rail_j: dict[str, float]
    ratio_wall_over_rail: float | None       # crude uncalibrated J_wall / J_rail — NOT a fit
    meter_resolution_wh: float | None        # smallest positive accumulator increment seen
    meter_cadence_s: float | None            # median seconds between accumulator updates
    n_samples: int


def probe(meter: EnergyMeter, source: MeterSource, *, seconds: float = 30.0,
          poll_s: float = 0.5, sleep=time.sleep, monotonic=time.monotonic) -> ProbeResult:
    """Poll the rail meter and the wall meter on one clock for `seconds`, then
    report synchronized deltas plus the plug's measured resolution/cadence. This
    is a diagnostic, not a calibration: it fits nothing and labels nothing."""
    t0 = monotonic()
    samples: list[tuple[float, float, EnergyByRail]] = []
    while True:
        t = monotonic() - t0
        samples.append((t, source.read_accumulated_wh(), meter.cumulative()))
        if t >= seconds:
            break
        sleep(poll_s)

    t_first, w_first, e_first = samples[0]
    t_last, w_last, e_last = samples[-1]
    dt = max(t_last - t_first, 1e-9)
    wall_wh = max(w_last - w_first, 0.0)
    wall_j = wall_wh * 3600.0
    rail = e_last - e_first                       # EnergyByRail.__sub__ clamps each rail ≥ 0
    rail_total_j = rail.total_j

    # characterize the accumulator from the times at which it actually changed
    change_ts = [samples[i][0] for i in range(1, len(samples))
                 if samples[i][1] > samples[i - 1][1]]
    increments = [samples[i][1] - samples[i - 1][1] for i in range(1, len(samples))
                  if samples[i][1] > samples[i - 1][1]]
    resolution = min(increments) if increments else None
    cadence = (median([change_ts[i] - change_ts[i - 1] for i in range(1, len(change_ts))])
               if len(change_ts) >= 2 else None)

    return ProbeResult(
        dt_s=dt, wall_wh=wall_wh, wall_w=wall_j / dt,
        rail_total_j=rail_total_j, rail_w=rail_total_j / dt,
        rail_by_rail_j=dict(rail.joules),
        ratio_wall_over_rail=(wall_j / rail_total_j) if rail_total_j > 0 else None,
        meter_resolution_wh=resolution, meter_cadence_s=cadence, n_samples=len(samples),
    )


def format_probe(r: ProbeResult) -> str:
    rails = "  ".join(f"{k}={v:.1f}J" for k, v in sorted(r.rail_by_rail_j.items()))
    ratio = f"{r.ratio_wall_over_rail:.3f}" if r.ratio_wall_over_rail is not None else "—"
    res = f"{r.meter_resolution_wh:.4f} Wh" if r.meter_resolution_wh is not None else "—"
    cad = f"{r.meter_cadence_s:.2f} s" if r.meter_cadence_s is not None else "—"
    return "\n".join([
        f"probe window: {r.dt_s:.1f}s, {r.n_samples} samples  (RAW diagnostic — not a calibration)",
        f"  wall (plug): {r.wall_wh:.4f} Wh   avg {r.wall_w:.2f} W",
        f"  rail (zeus): {r.rail_total_j:.1f} J   avg {r.rail_w:.2f} W   [{rails}]",
        f"  wall/rail ratio (J/J): {ratio}      ← should be stable & > 1 across runs if trustworthy",
        f"  plug resolution: {res}   update cadence: {cad}   ← pick C1 cell length well above this",
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/campaign.py tests/test_campaign.py
git commit -m "feat(calib): synchronized rail/wall probe + plug characterization"
```

---

### Task 5: `calibrate probe` CLI + construction seam

**Files:**
- Modify: `src/tokenwatt/campaign.py` (add `run_probe` construction seam)
- Modify: `src/tokenwatt/cli.py` (add the `calibrate` sub-app + `probe` command)
- Test: `tests/test_campaign.py` (run_probe seam), `tests/test_cli_smoke.py` (CLI wiring)

**Interfaces:**
- Consumes: `probe`, `ProbeResult`, `format_probe` (Task 4); `ShellyMeterSource` (Task 2); `ZeusMeter` (`tokenwatt.meter`); `load_config`, `ConfigError` (`tokenwatt.config`).
- Produces:
  - `run_probe(*, host, switch_id=0, password=None, seconds=30.0, poll_s=0.5, make_meter=<ZeusMeter factory>, make_source=<Shelly factory>, sleep=..., monotonic=...) -> tuple[ProbeResult | None, str]` — preflights reachability + meter construction, returning `(None, reason)` on failure.
  - CLI: `tokenwatt calibrate probe [--meter-host H] [--meter-id N] [--seconds S] [--poll P] [--config PATH]`.

- [ ] **Step 1: Write the failing test (run_probe seam)**

```python
# add to tests/test_campaign.py
from tokenwatt.meter import FakeMeter


def _ok_source(*_a, **_k):
    s = FakeMeterSource([1.0, 1.0, 1.001])
    s.reachable = lambda: (True, "fake ok")   # type: ignore[attr-defined]
    return s


def _dead_source(*_a, **_k):
    s = FakeMeterSource([1.0])
    s.reachable = lambda: (False, "ConnectError: no route")  # type: ignore[attr-defined]
    return s


def test_run_probe_returns_result_with_injected_fakes():
    clk = _Clock()
    result, msg = campaign.run_probe(
        host="h", seconds=1.0, poll_s=0.5,
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=_ok_source, sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert result.rail_total_j > 0
    assert "fake ok" in msg


def test_run_probe_aborts_loud_when_meter_unreachable():
    result, msg = campaign.run_probe(host="h", make_source=_dead_source,
                                     make_meter=lambda: FakeMeter())
    assert result is None                      # fail loud, no probe attempted
    assert "unreachable" in msg.lower()


def test_run_probe_degrades_when_rail_meter_unavailable():
    def _boom():
        raise RuntimeError("not Apple Silicon")
    result, msg = campaign.run_probe(host="h", make_source=_ok_source, make_meter=_boom)
    assert result is None
    assert "meter" in msg.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign.py -k run_probe -v`
Expected: FAIL — `AttributeError: module 'tokenwatt.campaign' has no attribute 'run_probe'`.

- [ ] **Step 3: Write minimal implementation (run_probe)**

```python
# add to src/tokenwatt/campaign.py (imports stay at top: time already imported)
def _default_meter():
    from tokenwatt.meter import ZeusMeter
    return ZeusMeter()


def _default_shelly(host: str, switch_id: int, password: str | None):
    from tokenwatt.metersource import ShellyMeterSource
    return ShellyMeterSource(host, switch_id=switch_id, password=password)


def run_probe(*, host: str, switch_id: int = 0, password: str | None = None,
              seconds: float = 30.0, poll_s: float = 0.5,
              make_meter=_default_meter, make_source=_default_shelly,
              sleep=time.sleep, monotonic=time.monotonic) -> tuple[ProbeResult | None, str]:
    """Construct the meters, preflight, and run `probe`. Returns (result, message);
    (None, reason) when the plug is unreachable or the rail meter is unavailable."""
    source = make_source(host, switch_id, password)
    ok, detail = source.reachable() if hasattr(source, "reachable") else (True, "")
    if not ok:
        return None, f"meter unreachable: {detail}"
    try:
        meter = make_meter()
    except Exception as e:
        return None, (f"energy meter unavailable ({type(e).__name__}: {e}); "
                      f"run this on the Apple-Silicon Mac being calibrated")
    result = probe(meter, source, seconds=seconds, poll_s=poll_s, sleep=sleep, monotonic=monotonic)
    return result, detail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign.py -k run_probe -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the failing test (CLI wiring)**

```python
# add to tests/test_cli_smoke.py
from typer.testing import CliRunner

from tokenwatt.cli import app

runner = CliRunner()


def test_calibrate_probe_help_lists_meter_host():
    res = runner.invoke(app, ["calibrate", "probe", "--help"])
    assert res.exit_code == 0
    assert "--meter-host" in res.output


def test_calibrate_probe_without_host_fails_loud():
    # no --meter-host and no config -> a pointable error, non-zero exit
    res = runner.invoke(app, ["calibrate", "probe"])
    assert res.exit_code == 1
    assert "meter host" in res.output.lower()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py -k calibrate -v`
Expected: FAIL — no such command `calibrate` (exit code 2 / usage error).

- [ ] **Step 7: Write minimal implementation (CLI)**

```python
# add to src/tokenwatt/cli.py, after the `app = typer.Typer(...)` line and callback

calibrate_app = typer.Typer(no_args_is_help=True,
                            help="Wall-meter calibration: connect a smart plug and relate zeus↔wall.")
app.add_typer(calibrate_app, name="calibrate")


@calibrate_app.command("probe")
def calibrate_probe(
    meter_host: Optional[str] = typer.Option(None, "--meter-host", help="Shelly Plus Plug US host/IP"),
    meter_id: int = typer.Option(0, "--meter-id", help="Gen2 Switch component id"),
    seconds: float = typer.Option(30.0, "--seconds", help="window length; generate load during it"),
    poll: float = typer.Option(0.5, "--poll", help="accumulator poll interval (s)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="read meter host from this config"),
):
    """RAW diagnostic: read synchronized rail-Δ (zeus) and wall-Δ (plug) over a
    window. Proves the plug data is trustworthy before any fitting. Not a calibration."""
    from tokenwatt import campaign
    from tokenwatt.config import load_config, ConfigError

    host, switch_id, password = meter_host, meter_id, None
    if host is None and config is not None:
        try:
            cfg = load_config(config)
        except ConfigError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        host = cfg.calibration.meter_host
        switch_id = cfg.calibration.meter_id
        password = cfg.calibration.meter_password
    if not host:
        typer.echo("no meter host — pass --meter-host or set calibration.meter_host in your config", err=True)
        raise typer.Exit(1)

    result, msg = campaign.run_probe(host=host, switch_id=switch_id, password=password,
                                     seconds=seconds, poll_s=poll)
    typer.echo(msg)
    if result is None:
        raise typer.Exit(1)
    typer.echo(campaign.format_probe(result))
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_smoke.py -k calibrate -v`
Expected: PASS (2 passed).

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new ones; no regressions.

- [ ] **Step 10: Commit**

```bash
git add src/tokenwatt/campaign.py src/tokenwatt/cli.py tests/test_campaign.py tests/test_cli_smoke.py
git commit -m "feat(calib): tokenwatt calibrate probe (synchronized rail/wall diagnostic)"
```

---

## On-device acceptance (run on the M3 Ultra, with the Shelly connected)

Not a CI step — the human/operator validation that closes C0 and feeds C1.

1. Put the plug host in `~/.tokenwatt/tokenwatt.yaml` under `calibration: { meter_host: <ip> }` (or pass `--meter-host`).
2. With the Mac idle: `tokenwatt calibrate probe --meter-host <ip> --seconds 20`. Note the idle wall W and the reported **plug resolution + cadence**.
3. Repeat under sustained load (e.g. a `decode`-style generation in another terminal). Confirm: wall W rises, rail J rises, and the **wall/rail ratio is > 1 and roughly stable** between two back-to-back runs.
4. Cross-check the wall Wh delta against the Shelly app's own energy reading for the same interval — they should agree.
5. **Record the measured cadence/resolution** — it sets the minimum C1 cell duration (windows must be well above the cadence so the accumulator delta dwarfs its resolution).

---

## Self-Review

**Spec coverage (C0 slice of `2026-06-30-tokenwatt-calibration-loop-design.md`):**
- §5 `MeterSource` protocol → Task 1; `ShellyMeterSource` (Gen2 RPC, `aenergy.total`, digest auth, `reachable()`) → Task 2; resolution/cadence caveat ("C0 measures it empirically") → Task 4 characterization + on-device step 5. `ManualMeterSource`/`LabMeterSource` explicitly deferred (documented).
- §14 C0 deliverables — protocol, Shelly source, plug host in config (Task 3), doctor-style reachability (`reachable()` in Task 2, surfaced by `run_probe` preflight in Task 5), `calibrate --probe` (Task 5, as the `calibrate probe` subcommand). *Naming note:* the spec wrote `calibrate --probe`; this plan ships it as the `calibrate probe` subcommand so `run`/`show` (C4) slot into the same typer group — a deliberate, documented deviation.
- §11 failure modes touched in C0 — Shelly unreachable (`reachable()` → fail loud), rail meter unavailable off-platform (`run_probe` degrades, doesn't crash).
- Honesty constraint — probe emits raw numbers + a labeled-uncalibrated ratio only; a test asserts the formatted output never contains `calibrated`.

**Placeholder scan:** none — every step ships real, runnable code and an exact command with expected output.

**Type consistency:** `read_accumulated_wh() -> float`, `reachable() -> tuple[bool,str]`, `probe(...) -> ProbeResult`, and `run_probe(...) -> tuple[ProbeResult|None, str]` are used identically wherever referenced; `FakeMeter(cumulative_step=...)` and `EnergyByRail.__sub__` match `meter.py`; `Config.calibration.meter_host/meter_id/meter_password` are consistent across Task 3 and Task 5.

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via executing-plans, batched with checkpoints.
