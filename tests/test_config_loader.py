import pytest
from tokenwatt.config import load_config, default_config, ConfigError


def test_default_config_has_catchall():
    c = default_config()
    assert c.routes[0].match == ["*"]
    assert c.routes[0].upstream == "http://127.0.0.1:8080"


def test_load_none_returns_default():
    assert load_config(None).routes[0].match == ["*"]


def test_load_valid_yaml(tmp_path):
    p = tmp_path / "tw.yaml"
    p.write_text(
        "port: 9000\n"
        "routes:\n"
        "  - name: m1\n"
        "    upstream: http://127.0.0.1:8080\n"
        "    match: [m1, 'mlx-community/Qwen3-*']\n"
    )
    c = load_config(str(p))
    assert c.port == 9000 and c.routes[0].name == "m1"
    assert c.routes[0].match == ["m1", "mlx-community/Qwen3-*"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_config(str(tmp_path / "nope.yaml"))
    assert "not found" in str(e.value)


def test_invalid_route_reports_field_path(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("routes:\n  - name: m1\n    upstream: not-a-url\n    match: [m1]\n")
    with pytest.raises(ConfigError) as e:
        load_config(str(p))
    msg = str(e.value)
    assert "routes" in msg and "upstream" in msg   # names the offending field path
