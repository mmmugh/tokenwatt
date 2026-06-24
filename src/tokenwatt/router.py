from __future__ import annotations
import fnmatch

from tokenwatt.config import RouteConfig


class Router:
    def __init__(self, routes: list[RouteConfig]) -> None:
        self._routes = routes

    def route_names(self) -> list[str]:
        return [r.name for r in self._routes]

    def upstreams(self) -> list[str]:
        """Distinct upstream base URLs, in first-seen (config) order."""
        seen: set[str] = set()
        out: list[str] = []
        for r in self._routes:
            if r.upstream not in seen:
                seen.add(r.upstream)
                out.append(r.upstream)
        return out

    def discoverable_upstreams(self) -> list[str]:
        """Distinct upstreams of routes opted into discovery (discover=True),
        in first-seen order. Excludes catalog-y backends (e.g. a vision server
        whose /v1/models lists the whole cache)."""
        seen: set[str] = set()
        out: list[str] = []
        for r in self._routes:
            if r.discover and r.upstream not in seen:
                seen.add(r.upstream)
                out.append(r.upstream)
        return out

    def upstream_type(self, upstream: str) -> str:
        """Type ('text'/'vision'/'embeddings') of the static route owning this
        upstream, so a dynamically-discovered route inherits the right modality.
        Defaults to 'text' if no static route declares the upstream."""
        for r in self._routes:
            if r.upstream == upstream:
                return r.type
        return "text"

    def discovered_route(self, model: str, upstream: str) -> RouteConfig:
        """Synthesize a route sending `model` to a live-discovered `upstream`."""
        return RouteConfig(name=f"discover:{model}", type=self.upstream_type(upstream),
                           upstream=upstream, match=[model])

    def resolve(self, model: str) -> RouteConfig | None:
        best_route: RouteConfig | None = None
        best_key = None
        for ri, route in enumerate(self._routes):
            for pi, pattern in enumerate(route.match):
                score = self._score(pattern, model)
                if score is None:
                    continue
                # earlier route / earlier pattern win ties -> negate indices so larger wins
                key = (score[0], score[1], -ri, -pi)
                if best_key is None or key > best_key:
                    best_key, best_route = key, route
        return best_route

    @staticmethod
    def _score(pattern: str, model: str):
        if pattern == "*":
            return (0, 0)                       # catch-all
        if "*" in pattern:
            if fnmatch.fnmatchcase(model, pattern):
                return (1, len(pattern.split("*", 1)[0]))   # glob; longer prefix = more specific
            return None
        if pattern == model:
            return (2, len(pattern))            # exact
        return None
