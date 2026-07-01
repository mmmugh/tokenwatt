from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


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


@dataclass(frozen=True)
class ShellyStatus:
    total_wh: float       # aenergy.total — the monotonic accumulator we integrate from
    apower_w: float       # instantaneous active power (sanity / live display only)
    voltage_v: float


class ShellyMeterSource:
    """Shelly smart-plug wall-energy meter over the Gen2+ RPC API (one API across
    Gen2/Gen3/Gen4; verified against the Plug US Gen4, model S4PL-00116US). Reads
    the plug's own accumulated-energy counter; no cloud, no sudo. The exact device
    model/gen is recorded into the calibration profile in C2."""
    name = "shelly-plug"
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

    def device_info(self) -> dict:
        base = self._host if "://" in self._host else f"http://{self._host}"
        r = self._client.get(f"{base}/rpc/Shelly.GetDeviceInfo", timeout=self._timeout)
        r.raise_for_status()
        b = r.json()
        return {"model": b.get("model"), "gen": b.get("gen"),
                "mac": b.get("mac"), "app": b.get("app")}

    def reachable(self) -> tuple[bool, str]:
        try:
            s = self.read_status()
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        return True, f"{self.name} @ {self._host}: {s.total_wh:.3f} Wh total, {s.apower_w:.1f} W now"

    def close(self) -> None:
        if self._own:
            self._client.close()
