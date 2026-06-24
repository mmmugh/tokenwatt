from __future__ import annotations
import asyncio
import time
from typing import Callable

import httpx


class Discovery:
    """Live ``model id -> upstream`` discovery so routing follows reality.

    Each backend (mlx-openai-server, mlx_vlm, LM Studio, ...) already exposes
    OpenAI ``/v1/models``. We poll every configured upstream, learn which model
    ids each currently serves, and cache the map. A request for a model is then
    routed to wherever that model is *actually loaded* — so swapping an mlx-tui
    slot or running ``lms load`` is picked up with no config edit and no proxy
    restart. Static routes remain the fallback (and supply the upstream list +
    per-upstream type).

    Freshness policy:
      * a cached map is "fresh" for ``ttl_s`` seconds;
      * a HIT on a fresh map returns immediately;
      * a MISS, or any lookup against a stale map, triggers a refresh — but no
        more often than every ``min_refresh_s`` (so a genuinely-absent model
        can't make every request hammer the backends);
      * refreshes are single-flight: concurrent callers coalesce onto one poll.
    """

    def __init__(self, *, client: httpx.AsyncClient, upstreams: Callable[[], list[str]],
                 ttl_s: float = 15.0, timeout_s: float = 0.8, min_refresh_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._client = client
        self._upstreams = upstreams
        self._ttl = ttl_s
        self._timeout = timeout_s
        self._min = min_refresh_s
        self._clock = clock
        self._map: dict[str, str] = {}
        self._last = float("-inf")        # monotonic time of the last successful poll
        self._lock = asyncio.Lock()

    def snapshot(self) -> dict[str, str]:
        """Current ``model -> upstream`` map (copy; for /status or debugging)."""
        return dict(self._map)

    async def upstream_for(self, model: str) -> str | None:
        now = self._clock()
        fresh = (now - self._last) < self._ttl
        if fresh and model in self._map:
            return self._map[model]
        # stale map, or a miss we're allowed to retry -> refresh (rate-limited, single-flight)
        if (not fresh) or (now - self._last) >= self._min:
            await self._refresh()
        return self._map.get(model)

    async def _refresh(self) -> None:
        async with self._lock:
            # Another coroutine may have refreshed while we waited on the lock.
            if (self._clock() - self._last) < self._min:
                return
            upstreams = list(self._upstreams())
            results = await asyncio.gather(*(self._poll(u) for u in upstreams))
            new_map: dict[str, str] = {}
            for upstream, ids in results:
                for mid in ids:
                    new_map.setdefault(mid, upstream)   # first upstream (config order) wins on duplicates
            self._map = new_map
            self._last = self._clock()

    async def _poll(self, upstream: str) -> tuple[str, list[str]]:
        """Return ``(upstream, [loaded model id, ...])``; empty on any error so one
        down backend never blocks the others.

        Backends disagree on what ``/v1/models`` means: mlx-openai-server lists
        exactly its served model (accurate), but LM Studio lists its whole on-disk
        CATALOG regardless of load state. So we prefer LM Studio's native
        ``/api/v0/models`` (which carries a load ``state``) and keep only
        ``state == 'loaded'``; plain OpenAI backends 404 there and fall back to
        ``/v1/models``."""
        lm = await self._get_json(f"{upstream}/api/v0/models")
        if lm is not None:
            data = self._models(lm)
            if any(isinstance(m, dict) and "state" in m for m in data):
                return upstream, [m["id"] for m in data
                                  if isinstance(m, dict) and m.get("id") and m.get("state") == "loaded"]
        oa = await self._get_json(f"{upstream}/v1/models")
        if oa is None:
            return upstream, []
        return upstream, [m["id"] for m in self._models(oa)
                          if isinstance(m, dict) and m.get("id")]

    async def _get_json(self, url: str):
        try:
            r = await self._client.get(url, timeout=self._timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _models(body) -> list:
        data = body.get("data") if isinstance(body, dict) else body
        return data if isinstance(data, list) else []
