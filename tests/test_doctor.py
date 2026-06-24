import json

import httpx

from tokenwatt import doctor as doc
from tokenwatt.config import Config, RouteConfig, ConfigError, DiscoveryConfig, RateConfig
from tokenwatt.meter import EnergyByRail


# ── tiny fakes ───────────────────────────────────────────────────────────────

class _RisingMeter:
    def __init__(self):
        self.j = 0.0

    def cumulative(self):
        self.j += 10.0
        return EnergyByRail({"cpu": self.j})


class _StaticMeter:
    def cumulative(self):
        return EnergyByRail({"cpu": 5.0})


def _cfg(routes=None, *, discovery=True, rate=0.31):
    return Config(routes=routes or [RouteConfig(name="mlx", upstream="http://m", match=["*"])],
                  discovery=DiscoveryConfig(enabled=discovery),
                  rate=RateConfig(flat_usd_per_kwh=rate))


def _backend_client(hosts):
    """hosts: name -> {"v1": [ids], "api": [(id, state)]}. Missing key -> 404/down."""
    def handler(request):
        host, path = request.url.host, request.url.path
        spec = hosts.get(host)
        if spec is None:
            raise httpx.ConnectError("down")
        if path == "/api/v0/models" and "api" in spec:
            return httpx.Response(200, json={"data": [{"id": i, "state": s} for i, s in spec["api"]]})
        if path == "/api/v0/models":
            return httpx.Response(404)
        if path == "/v1/models" and "v1" in spec:
            return httpx.Response(200, json={"data": [{"id": i} for i in spec["v1"]]})
        return httpx.Response(404)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── config ───────────────────────────────────────────────────────────────────

def test_config_ok_and_rate_warning():
    checks, cfg = doc.check_config("x", loader=lambda p: _cfg(rate=0.31))
    assert cfg is not None and [c.status for c in checks] == ["ok"]

    checks, cfg = doc.check_config("x", loader=lambda p: _cfg(rate=None))
    assert any(c.name == "rate" and c.status == "warn" for c in checks)


def test_config_parse_failure_is_fatal():
    def boom(_):
        raise ConfigError("invalid config in x:\n  routes.0.upstream: bad")
    checks, cfg = doc.check_config("x", loader=boom)
    assert cfg is None
    assert checks[0].status == "fail" and checks[0].fix


# ── ledger ───────────────────────────────────────────────────────────────────

def test_ledger_ok_in_writable_dir(tmp_path):
    checks = doc.check_ledger(str(tmp_path / "l.sqlite"))
    assert checks[0].status == "ok" and "0 request rows" in checks[0].detail


def test_ledger_missing_dir_fails_with_fix(tmp_path):
    checks = doc.check_ledger(str(tmp_path / "nope" / "l.sqlite"))
    assert checks[0].status == "fail" and checks[0].fix


# ── metering ─────────────────────────────────────────────────────────────────

def test_metering_ok_when_energy_moves():
    c = doc.check_metering(meter_factory=_RisingMeter, sleep=lambda _: None)
    assert c.status == "ok"


def test_metering_warns_when_flat():
    c = doc.check_metering(meter_factory=_StaticMeter, sleep=lambda _: None)
    assert c.status == "warn"


def test_metering_warns_when_meter_unavailable():
    def boom():
        raise RuntimeError("not apple silicon")
    c = doc.check_metering(meter_factory=boom, sleep=lambda _: None)
    assert c.status == "warn" and "unavailable" in c.detail


# ── proxy ────────────────────────────────────────────────────────────────────

def test_proxy_reachable_vs_not():
    assert doc.check_proxy("h", 7000, connect=lambda *_: True).status == "ok"
    assert doc.check_proxy("h", 7000, connect=lambda *_: False).status == "warn"


# ── upstreams + routing ──────────────────────────────────────────────────────

async def test_probe_distinguishes_loaded_vs_catalog():
    client = _backend_client({
        "mlx": {"v1": ["served-1"]},                                  # plain backend: lists its served model
        "lm": {"api": [("loaded-1", "loaded"), ("idle-1", "not-loaded")], "v1": ["loaded-1", "idle-1"]},
    })
    pm = await doc.probe_upstream(client, "http://mlx")
    assert pm.reachable and pm.served == ["served-1"] and not pm.is_lmstudio
    pl = await doc.probe_upstream(client, "http://lm")
    assert pl.is_lmstudio and pl.served == ["loaded-1"] and pl.catalog_size == 2   # idle filtered out


def test_catalog_backend_warns_unless_opted_out():
    cfg = _cfg(routes=[
        RouteConfig(name="vis", upstream="http://vis", match=["*VL*"], discover=False),
        RouteConfig(name="cat", upstream="http://cat", match=["*"]),
    ])
    probes = [doc.Probe("http://vis", True, served=["a", "b", "c"], catalog_size=3),
              doc.Probe("http://cat", True, served=["x", "y"], catalog_size=2)]
    checks = doc.check_upstreams_and_routing(cfg, probes)
    # opted-out vision: labeled, no warning
    vis = [c for c in checks if c.name == "http://vis"]
    assert len(vis) == 1 and vis[0].status == "ok" and "discovery off" in vis[0].detail
    # non-opted catalog backend: a warning with a fix
    assert any(c.name == "http://cat" and c.status == "warn" and c.fix for c in checks)


def test_routing_flags_collision():
    cfg = _cfg(routes=[
        RouteConfig(name="a", upstream="http://a", match=["x-only"]),
        RouteConfig(name="b", upstream="http://b", match=["*"]),
    ])
    probes = [doc.Probe("http://a", True, served=["dup"], catalog_size=1),
              doc.Probe("http://b", True, served=["dup"], catalog_size=1)]
    checks = doc.check_upstreams_and_routing(cfg, probes)
    assert any(c.name.startswith("collision:") and c.status == "warn" for c in checks)


def test_unreachable_upstream_warns():
    cfg = _cfg(routes=[RouteConfig(name="a", upstream="http://a", match=["*"])])
    checks = doc.check_upstreams_and_routing(cfg, [doc.Probe("http://a", reachable=False)])
    assert checks[0].status == "warn" and "unreachable" in checks[0].detail


# ── runner + reporting ───────────────────────────────────────────────────────

async def test_run_end_to_end_healthy(tmp_path):
    cfg = _cfg(routes=[RouteConfig(name="mlx", upstream="http://m", match=["*"])], rate=0.31)
    cfg.ledger = str(tmp_path / "l.sqlite")
    client = _backend_client({"m": {"v1": ["served-1"]}})
    checks = await doc._run_with(cfg, "cfg.yaml", fix=False, client=client,
                                 meter_factory=_RisingMeter, sleep=lambda _: None,
                                 connect=lambda *_: True)
    assert doc.exit_code(checks) == 0
    body = json.loads(doc.to_json(checks))
    assert body["ok"] is True and body["counts"]["fail"] == 0
    assert "served-1" in doc.format_text(checks) and "healthy" in doc.format_text(checks)


def test_exit_code_fail_unless_fixed():
    fail = doc.Check("ledger", "dir", "fail", "missing")
    assert doc.exit_code([fail]) == 1
    fail.fixed = True
    assert doc.exit_code([fail]) == 0
    assert doc.exit_code([doc.Check("x", "y", "warn")]) == 0   # warnings don't fail


def test_fix_creates_missing_ledger_dir(tmp_path):
    cfg = _cfg()
    cfg.ledger = str(tmp_path / "made" / "l.sqlite")
    checks = [doc.Check("ledger", "dir", "fail", "missing", fix="create dir")]
    doc._apply_fixes(checks, "cfg.yaml", cfg)
    assert checks[0].fixed and (tmp_path / "made").is_dir()
