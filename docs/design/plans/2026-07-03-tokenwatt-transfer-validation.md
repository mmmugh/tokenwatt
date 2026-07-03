# Transfer Validation — Battery-Aware Calibration + Per-Model Keying Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the code half of the transfer-validation phase — a sudoless battery-flux reader, a campaign battery gate, contaminated-cell exclusion in the fit, and per-(machine, model) profile keying — so the model×machine measurement sweep can run on laptops honestly.

**Architecture:** One new leaf module (`powersource.py`) mirroring `metersource.py`'s Source+Fake pattern; additive `battery_abs_w` fields threaded through `measure_idle`/`run_cell`/`run_campaign`; a fit-time exclusion + contaminated-idle guard in `calibration.py`; and an additive schema-2 (machine, model) key in `profiles.py` with legacy load. The byte-exact proxy path is untouched.

**Tech Stack:** Python ≥3.10, stdlib only for the new module (`subprocess`, `re`), pytest, `uv run pytest`.

## Global Constraints

- **No new runtime dependencies** — `powersource.py` is pure stdlib (`subprocess`, `re`).
- **Apple-Silicon / macOS only for the real reader**; it must **return `None` on desktops and non-Mac CI** (no battery / `ioreg` absent) so the suite runs anywhere on `FakeMeter`/fakes.
- **Honesty contract:** gate, never subtract; a contaminated idle baseline **fails loud** (no fake band); `battery_abs_w` is `None` when unknown, never a fabricated `0`.
- **Battery is calibration-only** — nothing here touches normal per-request metering.
- **Gate uses battery-power magnitude** `|InstantAmperage|·Voltage`; `IsCharging` is recorded but never gates. Threshold `GATE_W = 0.5` lives in `calibration.py`, applied at fit time.
- **Back-compat:** existing campaigns (no battery fields) and schema-1 profiles must keep working.
- `VERSION` auto-bumps its patch per commit via `.githooks/pre-commit` — expected; let it run.

---

### Task 1: `powersource.py` — battery-flux reader

**Files:**
- Create: `src/tokenwatt/powersource.py`
- Test: `tests/test_powersource.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `BatteryFlux(abs_w: float, charging: bool)` (frozen); `BatterySource` protocol with `read_flux(self) -> BatteryFlux | None`; `IOKitBatterySource(run=subprocess.check_output)`; `FakeBatterySource(fluxes: list[BatteryFlux | None])`. `read_flux()` returns `None` when there is no real battery (desktop) or `ioreg` is unreadable (CI).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_powersource.py
import pytest

from tokenwatt.powersource import (
    BatteryFlux, BatterySource, IOKitBatterySource, FakeBatterySource, _parse,
)

_CHARGING = '  "InstantAmperage" = 1500\n  "Voltage" = 12000\n  "IsCharging" = Yes\n'
# 2**64 - 1200 == InstantAmperage reported for -1200 mA (two's-complement wrap on discharge)
_DISCHARGING = '  "InstantAmperage" = 18446744073709550416\n  "Voltage" = 11400\n  "IsCharging" = No\n'
_NO_BATTERY = '  "InstantAmperage" = 0\n  "Voltage" = 0\n  "IsCharging" = No\n'


def test_parse_charging_uses_magnitude_and_flag():
    f = _parse(_CHARGING)
    assert f == BatteryFlux(abs_w=pytest.approx(18.0), charging=True)   # 1.5 A * 12 V


def test_parse_discharging_is_positive_magnitude_flag_false():
    f = _parse(_DISCHARGING)
    # magnitude only: |-1.2 A| * 11.4 V = 13.68 W; sign never enters the gate
    assert f.abs_w == pytest.approx(13.68) and f.charging is False


def test_parse_returns_none_when_no_real_battery():
    assert _parse(_NO_BATTERY) is None            # Voltage == 0 -> desktop


def test_parse_returns_none_on_missing_fields():
    assert _parse("nothing useful here") is None


def test_iokit_source_parses_injected_ioreg_output():
    def fake_run(args, text=True):
        assert args == ["ioreg", "-rn", "AppleSmartBattery"]
        return _CHARGING
    assert IOKitBatterySource(run=fake_run).read_flux() == BatteryFlux(18.0, True)


def test_iokit_source_returns_none_when_ioreg_absent():
    def boom(*a, **k):
        raise FileNotFoundError("ioreg")          # non-Mac CI / no such binary
    assert IOKitBatterySource(run=boom).read_flux() is None


def test_fake_source_replays_then_holds_last_and_satisfies_protocol():
    src = FakeBatterySource([BatteryFlux(20.0, True), None])
    assert isinstance(src, BatterySource)
    assert src.read_flux() == BatteryFlux(20.0, True)
    assert src.read_flux() is None
    assert src.read_flux() is None                # holds last; sustained reads never raise


def test_fake_source_rejects_empty():
    with pytest.raises(ValueError):
        FakeBatterySource([])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_powersource.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.powersource'`.

- [ ] **Step 3: Implement `powersource.py`**

```python
# src/tokenwatt/powersource.py
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BatteryFlux:
    """Instantaneous battery activity. `abs_w` is the MAGNITUDE of battery power
    (|InstantAmperage| × Voltage); `charging` is the IsCharging flag. The gate uses
    the magnitude, so the (empirically ambiguous) amperage sign never gates a cell."""
    abs_w: float
    charging: bool


@runtime_checkable
class BatterySource(Protocol):
    def read_flux(self) -> BatteryFlux | None: ...     # None == no battery (desktop)


def _parse(text: str) -> BatteryFlux | None:
    """Parse `ioreg -rn AppleSmartBattery` text. None when there is no real battery
    (Voltage == 0 or the field is absent) — the desktop / no-battery case."""
    def field(name: str) -> str | None:
        m = re.search(rf'"{name}"\s*=\s*(\S+)', text)
        return m.group(1) if m else None
    v, a = field("Voltage"), field("InstantAmperage")
    if v is None or a is None:
        return None
    voltage_mv = int(v)
    if voltage_mv == 0:
        return None
    amp = int(a)
    if amp >= 2 ** 63:                       # 64-bit two's-complement wrap -> negative (discharge)
        amp -= 2 ** 64
    abs_w = abs(amp) / 1000.0 * (voltage_mv / 1000.0)     # |mA|->A × mV->V
    return BatteryFlux(abs_w=abs_w, charging=(field("IsCharging") == "Yes"))


class IOKitBatterySource:
    """Reads the Mac's battery via `ioreg -rn AppleSmartBattery` (sudoless). Returns
    None on desktops (no battery) and anywhere `ioreg` is unavailable (non-Mac CI).
    Used ONLY to gate laptop calibration cells; never on the normal metering path."""
    def __init__(self, run=subprocess.check_output) -> None:
        self._run = run

    def read_flux(self) -> BatteryFlux | None:
        try:
            return _parse(self._run(["ioreg", "-rn", "AppleSmartBattery"], text=True))
        except Exception:
            return None                       # unreadable -> no signal (desktop/CI-safe)


class FakeBatterySource:
    """Deterministic test double: replays scripted BatteryFlux|None, holding the last."""
    def __init__(self, fluxes: list[BatteryFlux | None]) -> None:
        if not fluxes:
            raise ValueError("FakeBatterySource needs at least one flux")
        self._fluxes = list(fluxes)
        self._i = 0

    def read_flux(self) -> BatteryFlux | None:
        f = self._fluxes[min(self._i, len(self._fluxes) - 1)]
        self._i += 1
        return f
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_powersource.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/powersource.py tests/test_powersource.py
git commit -m "feat(calibration): sudoless battery-flux reader (powersource.py)"
```

---

### Task 2: campaign records `battery_abs_w` per idle + cell

**Files:**
- Modify: `src/tokenwatt/campaign.py` (`IdleRates`, `CellSample`, `measure_idle`, `run_cell`, `_default_battery`, `run_campaign`)
- Modify: `tests/test_campaign_run.py` (inject a fake battery into existing `run_campaign` calls; add new assertions)

**Interfaces:**
- Consumes: `BatterySource`, `BatteryFlux`, `FakeBatterySource` from Task 1.
- Produces: `IdleRates.battery_abs_w: float | None`; `CellSample.battery_abs_w: float | None`;
  `measure_idle(..., battery=None)`; `run_cell(..., battery=None, battery_poll_s=5.0)`;
  `run_campaign(..., make_battery=_default_battery)`. `battery_abs_w` is the max `abs_w`
  observed during the window, or `None` when no battery source / desktop.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_campaign_run.py
from tokenwatt.powersource import BatteryFlux, FakeBatterySource


def test_run_cell_records_max_battery_abs_w():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.0, 100.1])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    battery = FakeBatterySource([BatteryFlux(3.0, True), BatteryFlux(21.0, True)])  # peak 21 W
    def load_fn():
        clk.sleep(6.0)                                   # each call crosses a 5 s battery poll
        return ChatResult(tok_in=1, tok_out=1)
    s = campaign.run_cell(meter, source, load_fn, idle, cell="prefill", model="m",
                          seconds=12.0, battery=battery, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s.battery_abs_w == pytest.approx(21.0)         # the worst contamination in the cell


def test_run_cell_battery_none_when_no_source_or_desktop():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([1.0, 1.0])
    idle = campaign.IdleRates(rail_w={"cpu_total": 0.0}, wall_w=0.0, dt_s=1.0)
    def load_fn():
        clk.sleep(2.0)
        return ChatResult(tok_in=1, tok_out=1)
    # no source at all
    s1 = campaign.run_cell(meter, source, load_fn, idle, cell="c", model="m",
                           seconds=2.0, sleep=clk.sleep, monotonic=clk.monotonic)
    assert s1.battery_abs_w is None
    # desktop: source present but always yields None
    clk2 = _Clock()
    s2 = campaign.run_cell(meter, FakeMeterSource([1.0, 1.0]), load_fn, idle, cell="c",
                           model="m", seconds=2.0, battery=FakeBatterySource([None]),
                           sleep=clk2.sleep, monotonic=clk2.monotonic)
    assert s2.battery_abs_w is None


def test_measure_idle_records_battery_abs_w():
    clk = _Clock()
    meter = FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0}))
    source = FakeMeterSource([100.0, 100.05])
    idle = campaign.measure_idle(meter, source, seconds=10.0,
                                 battery=FakeBatterySource([BatteryFlux(9.0, True)]),
                                 sleep=clk.sleep, monotonic=clk.monotonic)
    assert idle.battery_abs_w == pytest.approx(9.0)


def test_run_campaign_threads_battery_into_samples():
    clk = _Clock()
    result, _ = campaign.run_campaign(
        cells=_cells(), model="m1",
        load=FakeLoadClient(tok_in=5, tok_out=9, latency_s=2.0, clock=clk),
        make_meter=lambda: FakeMeter(cumulative_step=EnergyByRail({"cpu_total": 10.0})),
        make_source=lambda *_a, **_k: FakeMeterSource([100.0 + 0.01 * i for i in range(50)]),
        make_battery=lambda: FakeBatterySource([BatteryFlux(4.0, True)]),
        host="h", cell_seconds=4.0, passes=1, timestamp=0.0,
        sleep=clk.sleep, monotonic=clk.monotonic)
    assert result is not None
    assert all(s.battery_abs_w == pytest.approx(4.0) for s in result.samples)
    assert result.idle.battery_abs_w == pytest.approx(4.0)
```

Also, in the **existing** `run_campaign` tests in this file
(`test_run_campaign_collects_one_sample_per_cell_per_pass`,
`test_run_campaign_emits_progress_per_phase`, `test_write_campaign_round_trips_json`),
add `make_battery=lambda: FakeBatterySource([None])` to each `run_campaign(...)` call to keep
them hermetic (no real `ioreg`). The three failure-path tests
(`..._meter_source_unreachable`, `..._meter_unavailable`, `..._read_failure_names_the_phase`)
abort before any battery read, so they need no change.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_campaign_run.py -q`
Expected: FAIL — `TypeError: run_cell() got an unexpected keyword argument 'battery'` (and the new fields absent).

- [ ] **Step 3: Implement the campaign changes**

In `src/tokenwatt/campaign.py`, add the field to `IdleRates` (after `dt_s`):

```python
@dataclass
class IdleRates:
    rail_w: dict[str, float]     # per-rail idle power (W)
    wall_w: float                # idle wall power (W)
    dt_s: float
    battery_abs_w: float | None = None   # max |battery power| during idle; None if no battery
```

Add the field to `CellSample` (after `requests`):

```python
    requests: int                         # count of load calls — proof sustained load ran
    battery_abs_w: float | None = None    # max |battery power| during the cell; None if no battery
```

Add the default-battery factory (next to `_default_meter`/`_default_shelly`):

```python
def _default_battery():
    from tokenwatt.powersource import IOKitBatterySource
    return IOKitBatterySource()
```

Replace `measure_idle` with the battery-aware version:

```python
def measure_idle(meter: EnergyMeter, source: MeterSource, *, seconds: float = 300.0,
                 battery=None, sleep=time.sleep, monotonic=time.monotonic) -> IdleRates:
    """Bracket an idle window (no inference) → per-rail idle watts + idle wall watts,
    plus the peak battery activity seen (to gate the baseline on laptops)."""
    t0 = monotonic()
    e0 = meter.cumulative()
    w0 = source.read_accumulated_wh()
    batt_max = _batt_peak(None, battery)
    sleep(seconds)
    dt = max(monotonic() - t0, 1e-9)
    e_rail = meter.cumulative() - e0
    wall_j = max(source.read_accumulated_wh() - w0, 0.0) * 3600.0
    batt_max = _batt_peak(batt_max, battery)
    return IdleRates(rail_w={r: j / dt for r, j in e_rail.joules.items()},
                     wall_w=wall_j / dt, dt_s=dt, battery_abs_w=batt_max)
```

Add the small helper (above `measure_idle`):

```python
def _batt_peak(cur: float | None, battery) -> float | None:
    """Read the battery source (if any) and fold its magnitude into the running max.
    Returns `cur` unchanged when there is no source or no battery (desktop)."""
    if battery is None:
        return cur
    f = battery.read_flux()
    if f is None:
        return cur
    return f.abs_w if cur is None else max(cur, f.abs_w)
```

Replace `run_cell` with the battery-aware version (only the loop + return change):

```python
def run_cell(meter: EnergyMeter, source: MeterSource, load_fn: Callable[[], ChatResult],
             idle: IdleRates, *, cell: str, model: str, seconds: float = 300.0,
             battery=None, battery_poll_s: float = 5.0,
             sleep=time.sleep, monotonic=time.monotonic) -> CellSample:
    """Drive `load_fn` in a loop for `seconds` while bracketing rail+wall energy and
    sampling battery activity ~every `battery_poll_s`; return the idle-subtracted sample."""
    t0 = monotonic()
    e0 = meter.cumulative()
    w0 = source.read_accumulated_wh()
    tok_in = tok_out = 0
    requests = 0
    any_unknown = False
    batt_max = None
    next_batt = t0
    while monotonic() - t0 < seconds:
        r = load_fn()
        requests += 1
        if r.tok_in is None or r.tok_out is None:
            any_unknown = True
        else:
            tok_in += r.tok_in
            tok_out += r.tok_out
        if battery is not None and monotonic() >= next_batt:
            batt_max = _batt_peak(batt_max, battery)
            next_batt = monotonic() + battery_poll_s
    dt = max(monotonic() - t0, 1e-9)
    e_rail = meter.cumulative() - e0
    wall_j = max(source.read_accumulated_wh() - w0, 0.0) * 3600.0
    rails = set(e_rail.joules) | set(idle.rail_w)
    marg_rail = {r: max(e_rail.joules.get(r, 0.0) - idle.rail_w.get(r, 0.0) * dt, 0.0)
                 for r in rails}
    marg_wall = max(wall_j - idle.wall_w * dt, 0.0)
    return CellSample(cell=cell, model=model, dt_s=dt, e_rail_marginal_j=marg_rail,
                      e_wall_marginal_j=marg_wall,
                      tok_in=None if (any_unknown or requests == 0) else tok_in,
                      tok_out=None if (any_unknown or requests == 0) else tok_out,
                      requests=requests, battery_abs_w=batt_max)
```

Thread it through `run_campaign` — add the parameter and pass `battery` down:

```python
def run_campaign(*, cells: list[LoadCell], model: str, load: LoadClient,
                 make_meter=_default_meter, make_source=_default_shelly,
                 make_battery=_default_battery,
                 host: str, switch_id: int = 0,
                 password: str | None = None, cell_seconds: float = 300.0, passes: int = 2,
                 timestamp: float, idle_seconds: float | None = None,
                 on_progress=lambda _m: None,
                 sleep=time.sleep, monotonic=time.monotonic) -> tuple["CampaignResult | None", str]:
```

Inside, after the meter is constructed and before `phase = "idle baseline"`, add:

```python
    battery = make_battery()
```

Pass `battery=battery` to the `measure_idle(...)` call and to the `run_cell(...)` call.

- [ ] **Step 4: Run to verify all campaign tests pass**

Run: `uv run pytest tests/test_campaign_run.py tests/test_campaign.py -q`
Expected: PASS (existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/campaign.py tests/test_campaign_run.py
git commit -m "feat(calibration): record per-cell battery activity in the campaign"
```

---

### Task 3: fit excludes battery-contaminated cells; fails loud on contaminated idle

**Files:**
- Modify: `src/tokenwatt/calibration.py` (`fit`, add `GATE_W`, `FitResult.n_excluded`)
- Modify: `src/tokenwatt/cli.py` (`calibrate_fit` — surface exclusion / catch `ValueError`)
- Test: `tests/test_calibration.py` (add exclusion + contaminated-idle tests)

**Interfaces:**
- Consumes: `campaign["samples"][i]["battery_abs_w"]`, `campaign["idle"]["battery_abs_w"]` from Task 2.
- Produces: `calibration.GATE_W = 0.5`; `FitResult.n_excluded: int` (default 0); `fit()` filters
  samples with `battery_abs_w >= GATE_W`, raises `ValueError` on a contaminated idle baseline or
  when no clean samples remain.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_calibration.py
import pytest
from tokenwatt import calibration


def _campaign(samples, idle_batt=None):
    return {
        "meter": {"tier": "smart_plug", "accuracy_pct": 1.0},
        "passes": 2,
        "idle": {"rail_w": {}, "wall_w": 0.0, "dt_s": 1.0, "battery_abs_w": idle_batt},
        "samples": samples,
    }


def _s(wall, rail, dt, cell="prefill", batt=None):
    return {"cell": cell, "dt_s": dt, "e_rail_marginal_j": {"gpu": rail},
            "e_wall_marginal_j": wall, "battery_abs_w": batt}


def test_fit_excludes_battery_contaminated_cells():
    clean = [_s(200.0, 100.0, 100.0, batt=0.1), _s(400.0, 200.0, 200.0, cell="decode", batt=0.2)]
    dirty = [_s(9999.0, 100.0, 100.0, cell="prefill", batt=25.0)]      # charging mid-cell
    r = calibration.fit(_campaign(clean + dirty))
    assert r.n_excluded == 1
    assert r.n_samples == 2                                            # only the clean cells fit


def test_fit_keeps_cells_with_no_battery_signal():
    # desktop / no-battery: battery_abs_w is None -> never excluded
    r = calibration.fit(_campaign([_s(200.0, 100.0, 100.0, batt=None),
                                   _s(400.0, 200.0, 200.0, cell="decode", batt=None)]))
    assert r.n_excluded == 0 and r.n_samples == 2


def test_fit_fails_loud_on_contaminated_idle_baseline():
    with pytest.raises(ValueError, match="idle baseline"):
        calibration.fit(_campaign([_s(200.0, 100.0, 100.0, batt=0.1)], idle_batt=12.0))


def test_fit_fails_loud_when_no_clean_samples():
    with pytest.raises(ValueError, match="no clean"):
        calibration.fit(_campaign([_s(9.0, 1.0, 1.0, batt=25.0)]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_calibration.py -q -k "battery or contaminated or clean"`
Expected: FAIL — `AttributeError: 'FitResult' object has no attribute 'n_excluded'` / no exclusion happens.

- [ ] **Step 3: Implement the fit changes**

In `src/tokenwatt/calibration.py`, add the constant near the top (after imports):

```python
GATE_W = 0.5     # a laptop cell is trusted only if |battery power| stayed below this (calibration only)
```

Add `n_excluded` to `FitResult` (default keeps every existing constructor valid):

```python
@dataclass(frozen=True)
class FitResult:
    fit_type: str
    a: float
    b: float
    residual_rel: float
    run_variance_rel: float
    band_pct: float
    tier: str
    n_samples: int
    n_passes: int
    n_excluded: int = 0
```

Add the gate helper and rewrite `fit`:

```python
def _contaminated(sample: dict) -> bool:
    b = sample.get("battery_abs_w")
    return b is not None and b >= GATE_W


def fit(campaign: dict) -> FitResult:
    idle_batt = (campaign.get("idle") or {}).get("battery_abs_w")
    if idle_batt is not None and idle_batt >= GATE_W:
        raise ValueError(
            f"idle baseline battery-contaminated ({idle_batt:.2f} W ≥ {GATE_W} W): the marginal "
            f"subtraction is unreliable — charge to 100% / cap charging and re-run")
    all_samples = campaign["samples"]
    samples = [s for s in all_samples if not _contaminated(s)]
    n_excluded = len(all_samples) - len(samples)
    if not samples:
        raise ValueError(
            "no clean samples to fit: every cell was battery-contaminated — "
            "charge to 100% / cap charging and re-run")
    a, b, residual_rel = fit_scalar(samples)
    rv = run_variance_rel(samples)
    acc = campaign["meter"]["accuracy_pct"]
    band = confidence_band_pct(residual_rel, rv, acc)
    return FitResult(
        fit_type="scalar", a=a, b=b, residual_rel=residual_rel, run_variance_rel=rv,
        band_pct=band, tier=tier_label(campaign["meter"]["tier"], band),
        n_samples=len(samples), n_passes=campaign.get("passes", 1), n_excluded=n_excluded)
```

In `src/tokenwatt/cli.py` `calibrate_fit`, widen the fit `try` to also surface the new guard, and
report exclusions. Replace the existing `try/except` around `calibration.fit`:

```python
    try:
        result = calibration.fit(camp)
    except RuntimeError as e:                     # nnls non-convergence — fail loud, not a traceback
        typer.echo(f"calibration did not converge: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:                       # battery-contaminated idle / no clean samples
        typer.echo(f"cannot fit: {e}", err=True)
        raise typer.Exit(1)
```

And after the `tier:` echo, add:

```python
    if result.n_excluded:
        typer.echo(f"note: excluded {result.n_excluded} battery-contaminated cell(s) from the fit")
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_calibration.py tests/test_cli_smoke.py -q`
Expected: PASS (new + existing; existing fit tests unaffected because campaigns without
`battery_abs_w` see `None` → never excluded, and idle without the field → no raise).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/calibration.py src/tokenwatt/cli.py tests/test_calibration.py
git commit -m "feat(calibration): exclude battery-contaminated cells; fail loud on dirty idle"
```

---

### Task 4: per-(machine, model) profile keying + model-gated lookup

**Files:**
- Modify: `src/tokenwatt/profiles.py` (`_PROFILE_SCHEMA` → 2, `_model_slug`, `save`, `load`, `active_for`, drop `active_for_current_machine`)
- Modify: `tests/test_profiles.py` (update signatures; add per-model + legacy + schema tests)

**Interfaces:**
- Consumes: `Profile.model_calibrated_on` (already present).
- Produces: profiles saved as `<machine_id>__<model_slug>.json`; `load(machine_id, model, *, root=None)`;
  `active_for(machine_id, model, *, root=None)`; schema-2 files; legacy schema-1 `<machine_id>.json`
  loads iff its `model_calibrated_on == model`; `load` raises `ValueError` on unknown `schema_version`.

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_profiles.py`'s save/load tests and add new ones:

```python
# tests/test_profiles.py  (replace test_save_then_load_round_trips, test_load_missing_machine_returns_none)
import json
import os
import pytest

from tokenwatt.profiles import Profile, profile_from, save, load, active_for, list_profiles
from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo


def _fit():
    return FitResult(fit_type="scalar", a=1.83, b=0.4, residual_rel=0.03,
                     run_variance_rel=0.02, band_pct=4.0, tier="plug-calibrated (±4%)",
                     n_samples=6, n_passes=2)


def _machine():
    return MachineInfo("mac15-14_apple-m3-ultra_96gb_macos26", "Apple M3 Ultra · 96 GB",
                       "Apple M3 Ultra", "Mac15,14", 96, "26")


def _profile(model):
    return profile_from(_fit(), _machine(),
                        meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                        model_calibrated_on=model, created_at=1_700_000_000.0)


def test_save_keys_by_machine_and_model(tmp_path):
    path = save(_profile("qwen3.6-27b"), root=str(tmp_path))
    assert path.endswith("mac15-14_apple-m3-ultra_96gb_macos26__qwen3-6-27b.json")
    got = load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path))
    assert got == _profile("qwen3.6-27b")


def test_two_models_on_one_machine_coexist(tmp_path):
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    save(_profile("gemma-4-e4b"), root=str(tmp_path))
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path)) is not None
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "gemma-4-e4b", root=str(tmp_path)) is not None
    assert len(list_profiles(root=str(tmp_path))) == 2


def test_load_missing_model_returns_none(tmp_path):
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    assert load("mac15-14_apple-m3-ultra_96gb_macos26", "other-model", root=str(tmp_path)) is None
    assert load("no-such-machine", "qwen3.6-27b", root=str(tmp_path)) is None


def test_legacy_schema1_file_loads_only_for_its_model(tmp_path):
    # a pre-existing <machine>.json (schema 1) still resolves for its own model
    legacy = _profile("qwen3.6-27b")
    from dataclasses import asdict
    d = asdict(legacy); d["schema_version"] = 1
    open(os.path.join(tmp_path, f"{legacy.machine_id}.json"), "w").write(json.dumps(d))
    assert load(legacy.machine_id, "qwen3.6-27b", root=str(tmp_path)) is not None
    assert load(legacy.machine_id, "different-model", root=str(tmp_path)) is None   # not a blind match


def test_load_raises_on_unknown_schema(tmp_path):
    p = save(_profile("qwen3.6-27b"), root=str(tmp_path))
    d = json.load(open(p)); d["schema_version"] = 99
    open(p, "w").write(json.dumps(d))
    with pytest.raises(ValueError, match="schema"):
        load("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path))


def test_active_for_is_exact_model_match(tmp_path):
    save(_profile("qwen3.6-27b"), root=str(tmp_path))
    assert active_for("mac15-14_apple-m3-ultra_96gb_macos26", "qwen3.6-27b", root=str(tmp_path)) is not None
    assert active_for("mac15-14_apple-m3-ultra_96gb_macos26", "gpt-oss-120b", root=str(tmp_path)) is None
```

Keep the existing `test_profile_from_carries_fit_machine_and_meter` unchanged.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_profiles.py -q`
Expected: FAIL — `ImportError: cannot import name 'active_for'` and `load()` signature mismatch.

- [ ] **Step 3: Implement the keying changes**

In `src/tokenwatt/profiles.py`, bump the schema and add a slug helper:

```python
_PROFILE_SCHEMA = 2
```

```python
import re

def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
```

Replace `save`, `load`, and `active_for_current_machine`:

```python
def save(profile: Profile, *, root: str | None = None) -> str:
    d = _root(root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{profile.machine_id}__{_model_slug(profile.model_calibrated_on)}.json")
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return path


def _read(path: str) -> Profile:
    with open(path) as f:
        data = json.load(f)
    sv = data.get("schema_version")
    if sv not in (1, 2):
        raise ValueError(f"unknown profile schema_version {sv!r} in {path}")
    return Profile(**data)


def load(machine_id: str, model: str, *, root: str | None = None) -> Profile | None:
    d = _root(root)
    keyed = os.path.join(d, f"{machine_id}__{_model_slug(model)}.json")
    if os.path.isfile(keyed):
        return _read(keyed)
    legacy = os.path.join(d, f"{machine_id}.json")           # pre-schema-2 single-model file
    if os.path.isfile(legacy):
        p = _read(legacy)
        if p.model_calibrated_on == model:
            return p
    return None


def active_for(machine_id: str, model: str, *, root: str | None = None) -> Profile | None:
    return load(machine_id, model, root=root)
```

Update `list_profiles` to skip any legacy `<machine>.json` double-count only if a keyed file
supersedes it — but for correctness here, `list_profiles` should just read every `*.json`; keep it
as-is (it already reads every `.json`, so both keyed and legacy files list). No change needed unless a
duplicate appears in practice; a legacy file with no keyed twin is a real profile and should list.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_profiles.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm no other caller broke, then run the whole suite**

```bash
grep -rn "active_for_current_machine\|profiles.load(" src/ tests/
uv run pytest -q
```
Expected: the grep shows no surviving references to the removed `active_for_current_machine` (the
CLI `calibrate_fit` only uses `profiles.save`, which now keys by model automatically); full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/tokenwatt/profiles.py tests/test_profiles.py
git commit -m "feat(calibration): key profiles by (machine, model) with legacy load"
```

---

## Post-build (not code tasks — run live with Justin, like the vardur campaign)

- **D — second-plug characterization:** confirm it is a metering Shelly, `tokenwatt calibrate probe`
  it, record accuracy/cadence/identity; assign it to the Air.
- **E — the sweep:** Ultra models (qwen3.6-27b ✓, qwen3.6-35B-A3B, qwen3-coder-next, gpt-oss-120b)
  at light 2 durations × 2 passes; then Qwen3.5-4B + Gemma-4-E4B on both Ultra and Air (battery-gated,
  residency-checked). Verify the Air's battery sign/`IsCharging` mapping on its first run; tune `GATE_W`.
- **Analysis & verdict:** do the per-model (a, b) cluster within band (→ per-machine scalar suffices) or
  spread (→ per-(machine, model) keying required)? Write the verdict; it feeds C3 (runtime apply, which
  consumes `active_for(machine_id, model)` + the model-gated band).

## Self-review notes

- **Spec coverage:** §5 → Task 1; §6 → Task 2; §7 (keying) → Task 4, (model-gated band + fail-loud
  schema) → Task 4; the fit-exclusion half of §6 → Task 3; §8/§9/§10 → Post-build. All covered.
- **Refinements vs spec (flagged to Justin):** gate on magnitude not signed flux (Task 1); threshold
  applied at fit time not in `run_cell` (Task 3); contaminated idle fails the whole fit (Task 3).
- **Type consistency:** `battery_abs_w: float | None` is identical across `IdleRates`, `CellSample`,
  the campaign dict, and `_contaminated`; `GATE_W` defined once in `calibration.py`; `load`/`active_for`
  share the `(machine_id, model)` signature.
