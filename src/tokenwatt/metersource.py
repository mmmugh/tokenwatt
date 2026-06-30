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
