import asyncio
import time
from importlib.resources import files
from typing import Optional

import typer
import httpx

from tokenwatt import __version__, cloud
from tokenwatt.ledger import Ledger
from tokenwatt.rate import FlatRate
from tokenwatt.idle import IdleBaseline

# Loaded from in-package data so it resolves for BOTH editable and wheel installs
# (a repo-root examples/ dir is NOT in the wheel and would FileNotFoundError at import).
EXAMPLE_CONFIG = files("tokenwatt").joinpath("data/m1-v1-embeddings.yaml").read_text(encoding="utf-8")


def _effective_log(cli_level, cli_file, cfg_logging):
    import os
    # precedence: CLI > env > config. Spec: only LEVEL has an env override
    # (TOKENWATT_LOG_LEVEL), not file. An empty/unset env falls through to config.
    level = cli_level or os.environ.get("TOKENWATT_LOG_LEVEL") or cfg_logging.level
    file = cli_file or cfg_logging.file
    return level, file

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _version_cb(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True),
):
    """TokenWatt — local-inference electricity cost meter."""


def _usd(x: float | None) -> str:
    if x is None:
        return "—"
    if 0 < x < 0.0001:
        return "<$0.0001"          # real, below the 4-decimal floor — never a fake $0
    return f"${x:.4f}"


def _kwh(x: float) -> str:          # no None branch: ledger kWh is always COALESCE'd to 0
    if 0 < x < 0.0001:
        return "<0.0001"
    return f"{x:.4f}"


def _model_label(m: str) -> str:
    """Display label: drop a namespace prefix (e.g. 'mlx-community/Foo-8bit' -> 'Foo-8bit')."""
    return m.rsplit("/", 1)[-1]


def _pmtok(v: float | None) -> str:
    """$/Mtok display (define near _usd): '-' when unknown, '<0.001' for a tiny positive —
    never collapse a real positive value to a fake '0.000'."""
    if v is None:
        return "-"
    return "<0.001" if 0 < v < 0.001 else f"{v:.3f}"


def render_report(ledger: Ledger, now: float) -> str:
    day = ledger.totals(now - 86_400)
    month = ledger.totals(now - 30 * 86_400)
    lines = [
        "TokenWatt — electricity cost of local inference",
        "  (numbers are ESTIMATED until you calibrate against a wall meter)",
        f"  last 24h : {day['requests']:>6} req   {_kwh(day['kwh'])} kWh   {_usd(day['usd'])}",
        f"  last 30d : {month['requests']:>6} req   {_kwh(month['kwh'])} kWh   {_usd(month['usd'])}",
    ]
    _ml = ledger.model_load_summary()
    if _ml["count"]:
        lines.append(f"  model loads: {_ml['count']} (booked separately: {_ml['total_load_j'] / 3.6e6 * 1000:.3f} Wh)")
    lines += [
        "",
        f"  {'model':<30}{'type':>11}{'req':>6}{'kWh':>12}{'$':>10}{'J/tok':>10}{'$/Mtok':>10}",
    ]
    for r in ledger.by_model():
        jpt = f"{r['j_per_token']:.3f}" if r["j_per_token"] is not None else "-"
        pm = _pmtok(r["usd_per_mtok"])
        lines.append(
            f"  {_model_label(r['model']):<30}{r['req_type']:>11}{r['requests']:>6}{_kwh(r['total_kwh']):>12}"
            f"{_usd(r['total_usd']):>10}{jpt:>10}{pm:>10}"
        )
    return "\n".join(lines)


@app.command()
def report(ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger")):
    """Show today/month electricity cost and a per-model breakdown."""
    import os
    path = os.path.expanduser(ledger)
    typer.echo(render_report(Ledger(path), now=time.time()))


@app.command()
def serve(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="path to tokenwatt.yaml"),
    upstream: Optional[str] = typer.Option(None, "--upstream", help="single-backend shortcut (overrides config routes)"),
    port: Optional[int] = typer.Option(None, "--port"),
    host: Optional[str] = typer.Option(None, "--host"),
    rate: Optional[float] = typer.Option(None, "--rate", help="flat $/kWh; omit to label costs 'estimated'"),
    ledger: Optional[str] = typer.Option(None, "--ledger"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
):
    """Run the measuring proxy (no sudo)."""
    import os
    from contextlib import asynccontextmanager
    import uvicorn
    from tokenwatt.config import load_config, RouteConfig, ConfigError
    from tokenwatt.router import Router
    from tokenwatt.meter import ZeusMeter
    from tokenwatt.proxy import create_app
    from tokenwatt.coldstart import ColdStartDetector

    try:
        cfg = load_config(config)
    except ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if upstream is not None:                       # single-backend shortcut
        cfg.routes = [RouteConfig(name="default", type="text", upstream=upstream, match=["*"])]
    if rate is not None:
        cfg.rate.flat_usd_per_kwh = rate

    from tokenwatt.log import setup_logging
    _lvl, _file = _effective_log(log_level, log_file, cfg.logging)
    setup_logging(level=_lvl, file=_file, console=cfg.logging.console,
                  max_bytes=cfg.logging.max_bytes, backup_count=cfg.logging.backup_count)

    eff_host = host if host is not None else cfg.host
    eff_port = port if port is not None else cfg.port
    eff_ledger = os.path.expanduser(ledger if ledger is not None else cfg.ledger)

    os.makedirs(os.path.dirname(eff_ledger), exist_ok=True)
    led = Ledger(eff_ledger)
    meter = ZeusMeter()
    idle = IdleBaseline(meter)
    # read=1800s: token generation can have long gaps on a loaded box; a tight read timeout
    # guillotines slow large-tool-call streams mid-generation (observed: ReadTimeout at 308s).
    # Generous + finite so a half-open upstream still eventually fails (and is logged).
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=1800.0, write=30.0, pool=10.0))
    detector = ColdStartDetector()
    inference_lock = asyncio.Lock() if cfg.serialize_inference else None

    @asynccontextmanager
    async def lifespan(app):
        # Establish an idle baseline BEFORE serving, so the first request isn't measured
        # against an empty baseline (a watt reading needs two cumulative reads ~0.5s apart).
        for _ in range(4):
            idle.sample()
            await asyncio.sleep(0.5)

        async def _idle_loop():
            while True:
                idle.sample()
                await asyncio.sleep(2.0)
        task = asyncio.create_task(_idle_loop())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.aclose()

    router = Router(cfg.routes)
    discovery = None
    if cfg.discovery.enabled:
        from tokenwatt.discovery import Discovery
        discovery = Discovery(client=client, upstreams=router.discoverable_upstreams,
                              ttl_s=cfg.discovery.ttl_s, timeout_s=cfg.discovery.timeout_s,
                              min_refresh_s=cfg.discovery.min_refresh_s)
    app_asgi = create_app(router=router, meter=meter, idle=idle, ledger=led,
                          rate=FlatRate(cfg.rate.flat_usd_per_kwh), client=client,
                          detector=detector, serialize_lock=inference_lock,
                          discovery=discovery, lifespan=lifespan)

    routes_desc = ", ".join(f"{r.name}->{r.upstream}" for r in cfg.routes) or "(none)"
    mode = "serialized" if cfg.serialize_inference else "concurrent"
    routing = "discover+static" if cfg.discovery.enabled else "static"
    typer.echo(f"TokenWatt proxy on http://{eff_host}:{eff_port}  routes: {routes_desc}  ({mode}, {routing}, no sudo)")
    uvicorn.run(app_asgi, host=eff_host, port=eff_port, log_level="warning")


@app.command()
def init(
    config: str = typer.Option("tokenwatt.yaml", "--config", "-c"),
    force: bool = typer.Option(False, "--force"),
):
    """Scaffold a commented tokenwatt.yaml you can edit."""
    import os
    if os.path.exists(config) and not force:
        typer.echo(f"{config} already exists; pass --force to overwrite", err=True)
        raise typer.Exit(1)
    with open(config, "w") as f:
        f.write(EXAMPLE_CONFIG)
    typer.echo(f"wrote {config}")


_GATE = 1.5   # outside the ±~30% measurement band before claiming a winner (cheaper/pricier vs comparable)


def _cloud_verdict(c: dict, local: float | None) -> str:
    """Three-way uncertainty-gated verdict vs a cloud total cost — shared by `compare` and
    `wrap_card` so the two can't drift. c = cloud.compare_total(...); ratio = cloud/local."""
    ratio = c["ratio"]
    if ratio >= _GATE:
        return f"~{ratio:.1f}× cheaper than {c['cloud']} ({_usd(c['cloud_usd'])} vs {_usd(local)})"
    if ratio <= 1 / _GATE:
        return f"~{1 / ratio:.1f}× pricier than {c['cloud']} ({_usd(c['cloud_usd'])} vs {_usd(local)})"
    return f"comparable to {c['cloud']} list ({_usd(c['cloud_usd'])} vs {_usd(local)}, within measurement uncertainty)"


def wrap_card(ledger: Ledger, now: float, days: int = 30) -> str:
    t = ledger.totals(now - days * 86_400)
    rows = ledger.by_model()
    tot_in = sum(r["total_in"] for r in rows)
    tot_out = sum(r["total_out"] for r in rows)
    local_total = t["usd"]
    lines = [
        f"## My local inference — last {days} days",
        f"- {t['requests']} requests · {_kwh(t['kwh'])} kWh · {_usd(local_total)} electricity",
        f"- {tot_in:,} input + {tot_out:,} output tokens metered",
    ]
    for r in rows:
        if r["usd_per_mtok"] is not None:
            lines.append(f"- {_model_label(r['model'])} ({r['req_type']}): "
                         f"${_pmtok(r['usd_per_mtok'])}/Mtok output · {r['requests']} req")
    share = "I metered my local LLM electricity with TokenWatt."
    c = cloud.compare_total(local_total, tot_in, tot_out)
    if c is not None:
        lines.append(f"- **vs cloud:** {_cloud_verdict(c, local_total)}")
        if c["ratio"] >= _GATE:
            share = (f"{days} days of local inference: {t['requests']} requests, {_usd(local_total)} of electricity. "
                     f"The same tokens would list at ~{_usd(c['cloud_usd'])} on {c['cloud']} (~{c['ratio']:.1f}×). "
                     f"Estimated ±15-30% pre-calibration. via TokenWatt")
    lines += [
        "",
        "_electricity estimated (±15-30%), pre wall-meter calibration; cloud = dated list price "
        "(incl. their compute + margin, not just power). $/Mtok output uses completion tokens "
        "incl. reasoning/<think>._",
        "",
        f"Share: {share}",
    ]
    return "\n".join(lines)


@app.command()
def compare(ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger")):
    """Compare your measured electricity against cloud LIST price for the same tokens."""
    import os
    led = Ledger(os.path.expanduser(ledger))
    typer.echo(f"local electricity vs cloud LIST price for the SAME tokens "
               f"(cloud snapshot {cloud.AS_OF}; list price incl. their compute+margin, not just power; edit cloud.py):")
    for r in led.by_model():
        local = r["total_usd"]
        c = cloud.compare_total(local, r["total_in"], r["total_out"])
        label = _model_label(r["model"])
        if c is None:
            if local is None:
                typer.echo(f"  {label:<28} no priced electricity (set --rate)")
            elif (r["total_in"] + r["total_out"]) == 0:
                typer.echo(f"  {label:<28} {_usd(local)} electricity — no tokens to compare")
            else:
                typer.echo(f"  {label:<28} {_usd(local)} electricity — $0 rate, no comparison")
            continue
        typer.echo(f"  {label:<28} {_usd(local)} electricity — {_cloud_verdict(c, local)}")


@app.command()
def wrap(ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger"),
         days: int = typer.Option(30, "--days")):
    """Print a shareable markdown card of your local-inference cost."""
    import os
    typer.echo(wrap_card(Ledger(os.path.expanduser(ledger)), now=time.time(), days=days))


def main():
    app()
