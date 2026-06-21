from __future__ import annotations
import fnmatch

from tokenwatt.config import RouteConfig


class Router:
    def __init__(self, routes: list[RouteConfig]) -> None:
        self._routes = routes

    def route_names(self) -> list[str]:
        return [r.name for r in self._routes]

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
