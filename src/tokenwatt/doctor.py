"""`tokenwatt doctor` — diagnose a TokenWatt setup and (optionally) apply safe
fixes. Checks the config, the proxy, every upstream, routing, the ledger, and
the no-sudo energy meter, then prints a grouped report (or --json) and exits
non-zero if anything is broken.

Every check is dependency-injected (config loader, http client, meter factory,
clock) so the whole thing is testable without real network or Apple-Silicon
hardware.
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Literal

import httpx

from tokenwatt.config import Config, ConfigError, load_config
from tokenwatt.router import Router

Status = Literal["ok", "warn", "fail"]
_SYMBOL = {"ok": "✓", "warn": "⚠", "fail": "✗"}


@dataclass
class Check:
    area: str
    name: str
    status: Status
    detail: str = ""
    fix: str | None = None      # one-line description of an available fix
    fixed: bool = False         # set when --fix applied it


@dataclass
class Probe:
    upstream: str
    reachable: bool
    served: list[str] = field(default_factory=list)   # what's actually loaded/served
    catalog_size: int = 0                              # entries in /v1/models (LM Studio/vlm list a catalog)
    is_lmstudio: bool = False                          # exposes /api/v0/models with a load `state`


def _models(body) -> list:
    data = body.get("data") if isinstance(body, dict) else body
    return data if isinstance(data, list) else []


async def probe_upstream(client: httpx.AsyncClient, upstream: str, timeout: float = 1.0) -> Probe:
    """What is this upstream actually serving? Prefers LM Studio's native
    /api/v0/models (load state); falls back to /v1/models for plain backends."""
    p = Probe(upstream, reachable=False)
    try:
        r = await client.get(f"{upstream}/api/v0/models", timeout=timeout)
        if r.status_code == 200:
            data = _models(r.json())
            if any(isinstance(m, dict) and "state" in m for m in data):
                p.reachable = p.is_lmstudio = True
                p.catalog_size = len(data)
                p.served = [m["id"] for m in data
                            if isinstance(m, dict) and m.get("id") and m.get("state") == "loaded"]
    except Exception:
        pass
    try:
        r = await client.get(f"{upstream}/v1/models", timeout=timeout)
        if r.status_code == 200:
            data = _models(r.json())
            p.reachable = True
            p.catalog_size = max(p.catalog_size, len(data))
            if not p.is_lmstudio:               # plain backend: /v1/models == served
                p.served = [m["id"] for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:
        pass
    return p


# ── individual checks ────────────────────────────────────────────────────────

def check_config(path: str, *, loader: Callable[[str], Config] = load_config) -> tuple[list[Check], Config | None]:
    try:
        cfg = loader(path)
    except ConfigError as e:
        first = str(e).splitlines()[0]
        return [Check("config", "parse", "fail", first,
                      fix="run `tokenwatt init` to scaffold a valid config")], None
    checks = [Check("config", "load", "ok",
                    f"{len(cfg.routes)} route(s); discovery {'on' if cfg.discovery.enabled else 'off'}")]
    if cfg.rate.flat_usd_per_kwh is None:
        checks.append(Check("config", "rate", "warn",
                            "rate.flat_usd_per_kwh unset → cost will report $0"))
    return checks, cfg


def check_ledger(path: str) -> list[Check]:
    expanded = os.path.expanduser(path)
    parent = os.path.dirname(expanded) or "."
    if not os.path.isdir(parent):
        return [Check("ledger", "dir", "fail", f"directory missing: {parent}",
                      fix=f"create {parent}")]
    if not os.access(parent, os.W_OK):
        return [Check("ledger", "dir", "fail", f"not writable: {parent}")]
    try:
        from tokenwatt.ledger import Ledger
        led = Ledger(expanded)            # constructing applies the schema
        with led._conn() as c:
            n = c.execute("SELECT count(*) FROM requests").fetchone()[0]
    except Exception as e:
        return [Check("ledger", "open", "fail", f"{type(e).__name__}: {e}")]
    return [Check("ledger", "open", "ok", f"{expanded} ({n} request rows)")]


def check_metering(*, meter_factory: Callable[[], object] | None = None,
                   sleep: Callable[[float], None] = time.sleep, dwell: float = 0.4) -> Check:
    """Confirm the no-sudo energy meter produces a moving reading. WARN (not
    FAIL) when unavailable — TokenWatt still meters tokens, just not energy."""
    if meter_factory is None:
        from tokenwatt.meter import ZeusMeter
        meter_factory = ZeusMeter
    try:
        m = meter_factory()
    except Exception as e:
        return Check("metering", "meter", "warn",
                     f"energy meter unavailable ({type(e).__name__}); token metering still works")
    try:
        e0 = m.cumulative().total_j
        sleep(dwell)
        e1 = m.cumulative().total_j
    except Exception as e:
        return Check("metering", "meter", "warn", f"meter read failed: {type(e).__name__}: {e}")
    if e1 > e0:
        return Check("metering", "meter", "ok", f"energy accumulating (+{e1 - e0:.2f} J in {dwell:.1f}s, no sudo)")
    return Check("metering", "meter", "warn", "meter returned no energy movement (idle, or IOReport stalled)")


def check_proxy(host: str, port: int, *,
                connect: Callable[[str, int, float], bool] | None = None, timeout: float = 1.0) -> Check:
    if connect is None:
        connect = _tcp_open
    where = f"{host}:{port}"
    return (Check("proxy", "listening", "ok", f"reachable on {where}")
            if connect(host, port, timeout)
            else Check("proxy", "listening", "warn", f"nothing listening on {where} (proxy not running?)"))


def _tcp_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def check_upstreams_and_routing(cfg: Config, probes: list[Probe]) -> list[Check]:
    """Per-upstream reachability + catalog warnings, then cross-upstream routing
    checks (collisions, empty discovery)."""
    by_up = {p.upstream: p for p in probes}
    discover_off = {r.upstream for r in cfg.routes if not r.discover}
    checks: list[Check] = []

    for up in Router(cfg.routes).upstreams():
        p = by_up.get(up)
        if p is None or not p.reachable:
            checks.append(Check("upstream", up, "warn", "unreachable (down or not started)"))
            continue
        if cfg.discovery.enabled and up in discover_off:
            checks.append(Check("upstream", up, "ok",
                                f"reachable; discovery off for this route (/v1/models lists {p.catalog_size})"))
            continue
        served = ", ".join(p.served) if p.served else "(nothing loaded)"
        checks.append(Check("upstream", up, "ok",
                            f"{'LM Studio' if p.is_lmstudio else 'serving'}: {served}"))
        # a plain backend that lists more than it serves is a cache catalog -> opt it out
        if cfg.discovery.enabled and not p.is_lmstudio and p.catalog_size > 1:
            checks.append(Check("upstream", up, "warn",
                                f"/v1/models lists {p.catalog_size} models (looks like a catalog, not what's loaded)",
                                fix="set `discover: false` on this route"))

    if cfg.discovery.enabled:
        discoverable = Router(cfg.routes).discoverable_upstreams()
        if not discoverable:
            checks.append(Check("routing", "discovery", "warn", "discovery on but every route opts out (discover:false)"))
        # collision: one model served by 2+ discoverable upstreams
        serving: dict[str, list[str]] = {}
        for p in probes:
            if p.upstream in discoverable:
                for mid in p.served:
                    serving.setdefault(mid, []).append(p.upstream)
        for mid, ups in serving.items():
            if len(ups) > 1:
                checks.append(Check("routing", f"collision:{mid}", "warn",
                                    f"loaded on {len(ups)} upstreams ({', '.join(ups)}); first wins"))
    return checks


# ── runner / formatting ──────────────────────────────────────────────────────

async def run(config_path: str | None, *, fix: bool = False,
              client: httpx.AsyncClient | None = None,
              meter_factory: Callable[[], object] | None = None,
              sleep: Callable[[float], None] = time.sleep,
              connect: Callable[[str, int, float], bool] | None = None) -> list[Check]:
    cfg_checks, cfg = check_config(config_path)
    if cfg is None:                       # config broken -> nothing else is meaningful
        checks = list(cfg_checks)
        if fix:
            _apply_fixes(checks, config_path, None)
        return checks
    rest = await _run_with(cfg, config_path, fix=fix, client=client,
                           meter_factory=meter_factory, sleep=sleep, connect=connect)
    return list(cfg_checks) + rest


async def _run_with(cfg: Config, config_path: str | None, *, fix: bool = False,
                    client: httpx.AsyncClient | None = None,
                    meter_factory: Callable[[], object] | None = None,
                    sleep: Callable[[float], None] = time.sleep,
                    connect: Callable[[str, int, float], bool] | None = None) -> list[Check]:
    checks = [check_proxy(cfg.host, cfg.port, connect=connect)]
    checks.extend(check_ledger(cfg.ledger))
    checks.append(check_metering(meter_factory=meter_factory, sleep=sleep))

    own = client is None
    if own:
        client = httpx.AsyncClient()
    try:
        probes = await asyncio.gather(*(probe_upstream(client, up) for up in Router(cfg.routes).upstreams()))
    finally:
        if own:
            await client.aclose()
    checks.extend(check_upstreams_and_routing(cfg, list(probes)))

    if fix:
        _apply_fixes(checks, config_path, cfg)
    return checks


def _apply_fixes(checks: list[Check], config_path: str | None, cfg: Config | None) -> None:
    """Apply only the safe, non-destructive fixes. Reports what it did via
    Check.fixed; never touches the ledger contents or restarts anything."""
    for c in checks:
        if not c.fix:
            continue
        if c.area == "ledger" and c.name == "dir" and cfg is not None:
            parent = os.path.dirname(os.path.expanduser(cfg.ledger)) or "."
            try:
                os.makedirs(parent, exist_ok=True)
                c.fixed = True
            except Exception:
                pass
        elif c.area == "config" and config_path and not os.path.exists(os.path.expanduser(config_path)):
            try:
                from tokenwatt.cli import EXAMPLE_CONFIG
                with open(os.path.expanduser(config_path), "w") as f:
                    f.write(EXAMPLE_CONFIG)
                c.fixed = True
            except Exception:
                pass


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "fail" and not c.fixed for c in checks) else 0


def to_json(checks: list[Check]) -> str:
    counts = _counts(checks)
    return _json.dumps({"ok": exit_code(checks) == 0, "counts": counts,
                        "checks": [asdict(c) for c in checks]}, indent=2)


def format_text(checks: list[Check]) -> str:
    lines, area = [], None
    for c in checks:
        if c.area != area:
            area = c.area
            lines.append(f"\n{area}:")
        mark = _SYMBOL[c.status]
        tail = "  [fixed]" if c.fixed else (f"  → fix: {c.fix}" if c.fix else "")
        lines.append(f"  {mark} {c.name}: {c.detail}{tail}")
    co = _counts(checks)
    verdict = "healthy" if exit_code(checks) == 0 else "PROBLEMS"
    lines.append(f"\n{verdict} — {co['ok']} ok, {co['warn']} warn, {co['fail']} fail")
    return "\n".join(lines).lstrip("\n")


def _counts(checks: list[Check]) -> dict[str, int]:
    return {s: sum(1 for c in checks if c.status == s) for s in ("ok", "warn", "fail")}
