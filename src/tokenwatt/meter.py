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
