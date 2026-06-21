from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FlatRate:
    usd_per_kwh: float | None

    def price(self, kwh: float) -> float | None:
        if self.usd_per_kwh is None:
            return None
        return kwh * self.usd_per_kwh
