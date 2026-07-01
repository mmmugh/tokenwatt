# src/tokenwatt/machineid.py
from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass


def _real_sysctl(key: str) -> str:
    return subprocess.check_output(["sysctl", "-n", key], text=True).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@dataclass(frozen=True)
class MachineInfo:
    machine_id: str
    label: str
    soc: str
    model: str
    ram_gb: int
    macos_major: str


def detect_machine(*, sysctl=_real_sysctl, mac_ver=lambda: platform.mac_ver()[0]) -> MachineInfo:
    soc = sysctl("machdep.cpu.brand_string")
    model = sysctl("hw.model")
    ram_gb = round(int(sysctl("hw.memsize")) / 1024**3)
    macos_major = (mac_ver() or "0").split(".")[0]
    machine_id = f"{_slug(model)}_{_slug(soc)}_{ram_gb}gb_macos{macos_major}"
    label = f"{soc} · {ram_gb} GB · {model} · macOS {macos_major}"
    return MachineInfo(machine_id=machine_id, label=label, soc=soc, model=model,
                       ram_gb=ram_gb, macos_major=macos_major)
