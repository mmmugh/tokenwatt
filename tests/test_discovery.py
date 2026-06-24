import asyncio
import httpx

from tokenwatt.discovery import Discovery
from tokenwatt.config import RouteConfig
from tokenwatt.router import Router
from tokenwatt.proxy import create_app
from tokenwatt.meter import FakeMeter
from tokenwatt.idle import IdleBaseline
from tokenwatt.rate import FlatRate
from tokenwatt.ledger import Ledger
from tokenwatt.coldstart import ColdStartDetector


class _Clock:
    """Manually-advanced monotonic clock for deterministic TTL/refresh tests."""
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


_UPSTREAMS = ["http://up-a", "http://up-b", "http://up-down"]   # up-down has no registry -> errors


def _disc(registry: dict[str, list[str]], calls: dict[str, int], clock, *,
          lmstudio: dict[str, list[tuple[str, str]]] | None = None,
          upstreams: list[str] | None = None, **kw) -> Discovery:
    """Discovery wired to a MockTransport. mlx-style hosts (in `registry`) 404 on
    /api/v0/models and serve their loaded model on /v1/models. LM-Studio-style
    hosts (in `lmstudio`, host -> [(id, state)]) serve load state on
    /api/v0/models and a full catalog on /v1/models (which must be ignored).
    `calls` tallies the authoritative poll per host."""
    lmstudio = lmstudio or {}

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host in lmstudio:
            if path == "/api/v0/models":
                calls[host] = calls.get(host, 0) + 1
                return httpx.Response(200, json={"data": [{"id": i, "state": s} for i, s in lmstudio[host]]})
            return httpx.Response(200, json={"data": [{"id": i} for i, _ in lmstudio[host]]})  # catalog -> ignored
        if host not in registry:
            raise httpx.ConnectError("down")
        if path == "/api/v0/models":
            return httpx.Response(404)            # not LM Studio -> caller falls back to /v1/models
        calls[host] = calls.get(host, 0) + 1
        return httpx.Response(200, json={"object": "list",
                                         "data": [{"id": m, "object": "model"} for m in registry[host]]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ups = upstreams if upstreams is not None else _UPSTREAMS
    return Discovery(client=client, upstreams=lambda: list(ups), clock=clock, **kw)


# ── discovery unit behavior ──────────────────────────────────────────────────

async def test_maps_models_to_their_upstreams():
    d = _disc({"up-a": ["model-a"], "up-b": ["model-b1", "model-b2"]}, {}, _Clock())
    assert await d.upstream_for("model-a") == "http://up-a"
    assert await d.upstream_for("model-b2") == "http://up-b"
    assert await d.upstream_for("not-loaded") is None


async def test_fresh_hit_does_not_repoll():
    calls = {}
    d = _disc({"up-a": ["model-a"]}, calls, _Clock(), ttl_s=15.0)
    await d.upstream_for("model-a")
    n = calls["up-a"]
    await d.upstream_for("model-a")            # fresh hit -> no new poll
    assert calls["up-a"] == n


async def test_miss_picks_up_a_newly_loaded_model():
    calls, clock = {}, _Clock()
    reg = {"up-a": ["model-a"]}
    d = _disc(reg, calls, clock, ttl_s=15.0, min_refresh_s=2.0)
    assert await d.upstream_for("model-a") == "http://up-a"
    reg["up-a"].append("model-new")           # model swapped in on the backend
    clock.t += 3.0                            # past min_refresh so a miss re-polls
    assert await d.upstream_for("model-new") == "http://up-a"


async def test_stale_ttl_triggers_refresh_even_on_a_hit():
    calls, clock = {}, _Clock()
    reg = {"up-a": ["model-a"]}
    d = _disc(reg, calls, clock, ttl_s=15.0)
    await d.upstream_for("model-a")
    n = calls["up-a"]
    reg["up-a"] = ["model-a", "model-z"]
    clock.t += 20.0                           # past ttl
    await d.upstream_for("model-a")           # stale -> refresh
    assert calls["up-a"] == n + 1
    assert await d.upstream_for("model-z") == "http://up-a"


async def test_down_upstream_is_skipped_not_fatal():
    d = _disc({"up-a": ["model-a"]}, {}, _Clock())       # up-b, up-down error on every probe
    assert await d.upstream_for("model-a") == "http://up-a"   # a down peer didn't break up-a
    assert "http://up-down" not in d.snapshot().values()


async def test_lmstudio_uses_loaded_state_not_catalog():
    # LM Studio's /v1/models is a catalog; only /api/v0/models state=='loaded' is routable
    d = _disc({}, {}, _Clock(), upstreams=["http://up-a"],
              lmstudio={"up-a": [("m-loaded", "loaded"), ("m-idle", "not-loaded")]})
    assert await d.upstream_for("m-loaded") == "http://up-a"
    assert await d.upstream_for("m-idle") is None        # in catalog but not loaded -> not routable


async def test_min_refresh_rate_limits_repeated_misses():
    calls, clock = {}, _Clock()
    d = _disc({"up-a": ["model-a"]}, calls, clock, min_refresh_s=2.0, ttl_s=15.0)
    await d.upstream_for("model-a")
    n = calls["up-a"]
    assert await d.upstream_for("ghost") is None        # miss within min_refresh window
    assert calls["up-a"] == n                            # ... must NOT have re-polled


async def test_concurrent_first_lookups_coalesce_to_one_poll():
    calls = {}
    d = _disc({"up-a": ["model-a"]}, calls, _Clock())
    results = await asyncio.gather(*[d.upstream_for("model-a") for _ in range(12)])
    assert all(r == "http://up-a" for r in results)
    assert calls["up-a"] == 1                            # single-flight


async def test_first_upstream_wins_on_duplicate_model():
    # same model id served by two backends -> config (upstream) order wins
    d = _disc({"up-a": ["dup"], "up-b": ["dup"]}, {}, _Clock())
    assert await d.upstream_for("dup") == "http://up-a"


# ── router helpers ───────────────────────────────────────────────────────────

def test_discovered_route_inherits_upstream_type():
    router = Router([
        RouteConfig(name="vis", type="vision", upstream="http://v:8081", match=["*VL*"]),
        RouteConfig(name="txt", type="text", upstream="http://m:8080", match=["*"]),
    ])
    r = router.discovered_route("Qwen3-VL", "http://v:8081")
    assert (r.upstream, r.type, r.match) == ("http://v:8081", "vision", ["Qwen3-VL"])
    assert router.upstream_type("http://m:8080") == "text"
    assert router.upstream_type("http://unknown") == "text"   # default


def test_discoverable_upstreams_excludes_opted_out():
    router = Router([
        RouteConfig(name="vis", type="vision", upstream="http://v:8081", match=["*VL*"], discover=False),
        RouteConfig(name="lm", type="text", upstream="http://m:8080", match=["*"]),
    ])
    assert router.discoverable_upstreams() == ["http://m:8080"]   # vision opted out
    assert "http://v:8081" in router.upstreams()                  # still known (for /v1/models aggregation)


# ── proxy integration: discovery overrides static, then falls back ───────────

def _proxy(router, client, discovery, tmp_path):
    return create_app(router=router, meter=FakeMeter(), idle=IdleBaseline(FakeMeter()),
                      ledger=Ledger(str(tmp_path / "l.sqlite")), rate=FlatRate(0.31),
                      client=client, detector=ColdStartDetector(), discovery=discovery)


def _backend_client(live: dict[str, list[str]]) -> httpx.AsyncClient:
    """A host-aware fake backend reachable as http://a and http://b through one
    ASGI app (real streams, unlike MockTransport): /v1/models reports `live` for
    the requested host, chat echoes which host served the request."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def models(request):
        return JSONResponse({"data": [{"id": m} for m in live.get(request.url.hostname, [])]})

    async def chat(request):
        return JSONResponse({"served_by": request.url.hostname,
                             "choices": [{"message": {"content": "ok"}}],
                             "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    app = Starlette(routes=[
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
    ])
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app))


_TWO = [
    RouteConfig(name="b", type="text", upstream="http://b", match=["x-never"]),  # declares B as an upstream
    RouteConfig(name="catch", type="text", upstream="http://a", match=["*"]),    # static catch-all -> A
]


async def test_discovery_overrides_static_catchall(tmp_path):
    # static catch-all sends everything to A, but the model is actually live on B
    router = Router(list(_TWO))
    client = _backend_client({"a": [], "b": ["m-live"]})
    app = _proxy(router, client, Discovery(client=client, upstreams=router.upstreams), tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions", json={"model": "m-live"})
    assert r.json()["served_by"] == "b"          # discovery beat the static catch-all (a)


async def test_undiscovered_model_falls_back_to_static(tmp_path):
    router = Router(list(_TWO))
    client = _backend_client({"a": [], "b": []})   # nothing live anywhere
    app = _proxy(router, client, Discovery(client=client, upstreams=router.upstreams), tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tw") as c:
        r = await c.post("/v1/chat/completions", json={"model": "m-unknown"})
    assert r.json()["served_by"] == "a"          # fell back to the static catch-all
