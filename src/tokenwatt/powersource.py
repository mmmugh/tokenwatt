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
        # real-battery ioreg packs fields comma-separated with no spaces
        # ("Voltage" = 12590,"LifetimeData"=...) so stop the value at a comma/brace,
        # not just whitespace — otherwise int() sees the whole rest of the line.
        m = re.search(rf'"{name}"\s*=\s*([^,\s}}]+)', text)
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
