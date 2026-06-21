from tokenwatt.config import RouteConfig
from tokenwatt.router import Router


def _r(name, *match):
    return RouteConfig(name=name, upstream="http://x", match=list(match))


def test_exact_beats_glob_and_catchall():
    router = Router([_r("cap", "*"), _r("glob", "mlx-*"), _r("exact", "mlx-7b")])
    assert router.resolve("mlx-7b").name == "exact"


def test_glob_beats_catchall_and_longest_prefix_wins():
    router = Router([_r("cap", "*"), _r("short", "mlx-*"), _r("long", "mlx-community/*")])
    assert router.resolve("mlx-community/Qwen").name == "long"   # longer literal prefix
    assert router.resolve("mlx-7b").name == "short"


def test_first_in_list_breaks_ties():
    router = Router([_r("a", "mlx-*"), _r("b", "mlx-*")])
    assert router.resolve("mlx-7b").name == "a"


def test_catchall_matches_anything_and_none_when_unmatched():
    assert Router([_r("cap", "*")]).resolve("whatever").name == "cap"
    assert Router([_r("only", "m1")]).resolve("m2") is None
