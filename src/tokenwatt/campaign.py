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
