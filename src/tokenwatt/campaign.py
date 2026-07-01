# src/tokenwatt/campaign.py
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from statistics import median
from typing import Callable

from tokenwatt.battery import ChatResult, LoadCell, LoadClient
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


def format_probe(r: ProbeResult) -> str:
    rails = "  ".join(f"{k}={v:.1f}J" for k, v in sorted(r.rail_by_rail_j.items()))
    ratio = f"{r.ratio_wall_over_rail:.3f}" if r.ratio_wall_over_rail is not None else "—"
    if r.meter_resolution_wh is None or r.meter_cadence_s is None:
        # the window was too short to see the accumulator change >=2 times — bare dashes next
        # to "pick C1 cell length well above this" would be self-contradictory (nothing to
        # compare against), so print an actionable hint instead.
        cadence_line = ("  plug resolution/cadence: not observed — the window was shorter than "
                        "the plug's energy-counter update interval; re-run with a larger --seconds")
    else:
        res = f"{r.meter_resolution_wh:.4f} Wh"
        cad = f"{r.meter_cadence_s:.2f} s"
        cadence_line = f"  plug resolution: {res}   update cadence: {cad}   ← pick C1 cell length well above this"
    return "\n".join([
        f"probe window: {r.dt_s:.1f}s, {r.n_samples} samples  (RAW diagnostic — not a calibration)",
        f"  wall (plug): {r.wall_wh:.4f} Wh   avg {r.wall_w:.2f} W",
        f"  rail (zeus): {r.rail_total_j:.1f} J   avg {r.rail_w:.2f} W   [{rails}]",
        f"  wall/rail ratio (J/J): {ratio}      ← should be stable & > 1 across runs if trustworthy",
        cadence_line,
    ])


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
                      tok_in=None if (any_unknown or requests == 0) else tok_in,
                      tok_out=None if (any_unknown or requests == 0) else tok_out, requests=requests)


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
                 make_meter=_default_meter, make_source=_default_shelly,
                 host: str, switch_id: int = 0,
                 password: str | None = None, cell_seconds: float = 300.0, passes: int = 2,
                 timestamp: float, idle_seconds: float | None = None,
                 on_progress=lambda _m: None,
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
    phase = "idle baseline"
    try:
        on_progress(f"measuring {phase} ({idle_seconds or cell_seconds:.0f}s)…")
        idle = measure_idle(meter, source, seconds=idle_seconds or cell_seconds,
                            sleep=sleep, monotonic=monotonic)
        samples: list[CellSample] = []
        for p in range(passes):
            for cell in cells:
                phase = f"cell {cell.name} pass {p + 1}/{passes}"
                on_progress(f"{phase} ({cell_seconds:.0f}s)…")
                samples.append(run_cell(
                    meter, source, lambda c=cell: load.chat(model, c.prompt, c.max_tokens),
                    idle, cell=cell.name, model=model, seconds=cell_seconds,
                    sleep=sleep, monotonic=monotonic))
    except Exception as e:
        return None, f"campaign read failed during {phase} ({type(e).__name__}: {e})"
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
