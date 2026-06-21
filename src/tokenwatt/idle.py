from __future__ import annotations
import time
from collections import deque
from statistics import median

from tokenwatt.meter import EnergyByRail, EnergyMeter


class IdleBaseline:
    """Rolling per-rail idle power (watts) from cumulative-energy deltas, sampled only when
    no request is in flight (spec §7). `energy_over` uses the MEDIAN of the recent samples so
    a single noisy reading can't skew the baseline."""

    def __init__(self, meter: EnergyMeter, window: int = 8) -> None:
        self._meter = meter
        self._last: EnergyByRail | None = None
        self._last_t: float | None = None
        self._samples: deque[dict[str, float]] = deque(maxlen=window)
        self._in_flight: int = 0

    def request_started(self) -> None:
        self._in_flight += 1

    def request_finished(self) -> None:
        if self._in_flight > 0:
            self._in_flight -= 1

    def in_flight(self) -> int:
        return self._in_flight

    def sample(self) -> None:
        now = time.monotonic()
        cur = self._meter.cumulative()
        if self._in_flight > 0:
            # busy: record no idle sample, but ADVANCE the cursor so the next idle sample
            # measures only the post-inference gap, not across the busy span (which would
            # fold inference power into the 'idle' baseline). spec §7
            self._last, self._last_t = cur, now
            return
        if self._last is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                delta = cur - self._last
                self._samples.append({r: j / dt for r, j in delta.joules.items()})
        self._last, self._last_t = cur, now

    def energy_over(self, seconds: float) -> EnergyByRail:
        if not self._samples:
            return EnergyByRail({})
        rails = set().union(*(s.keys() for s in self._samples))
        return EnergyByRail({
            r: median([s.get(r, 0.0) for s in self._samples]) * seconds
            for r in rails
        })
