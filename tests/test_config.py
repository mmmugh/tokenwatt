import pytest
from pydantic import ValidationError
from tokenwatt.config import Config, RouteConfig, CalibrationConfig


def test_valid_config_parses_with_defaults():
    c = Config(routes=[{"name": "m1", "upstream": "http://127.0.0.1:8080/", "match": ["m1"]}])
    assert c.port == 7000 and c.host == "127.0.0.1"
    assert c.routes[0].type == "text"
    assert c.routes[0].upstream == "http://127.0.0.1:8080"   # trailing slash stripped


def test_bad_upstream_rejected():
    with pytest.raises(ValidationError) as e:
        RouteConfig(name="m1", upstream="127.0.0.1:8080", match=["m1"])
    assert "http://" in str(e.value)


def test_bad_type_rejected():
    with pytest.raises(ValidationError) as e:
        RouteConfig(name="m1", type="embedding", upstream="http://x", match=["m1"])  # missing 's'
    assert "embeddings" in str(e.value)   # message names the allowed values (helpful, fail-loud)


def test_empty_match_rejected():
    with pytest.raises(ValidationError):
        RouteConfig(name="m1", upstream="http://x", match=[])


def test_duplicate_route_names_rejected():
    with pytest.raises(ValidationError) as e:
        Config(routes=[
            {"name": "m1", "upstream": "http://a", "match": ["a"]},
            {"name": "m1", "upstream": "http://b", "match": ["b"]},
        ])
    assert "duplicate" in str(e.value).lower()


def test_calibration_defaults_are_empty():
    # zero-config must still construct; an absent plug is not an error
    cfg = Config()
    assert isinstance(cfg.calibration, CalibrationConfig)
    assert cfg.calibration.meter_host is None
    assert cfg.calibration.meter_id == 0
    assert cfg.calibration.meter_password is None


def test_calibration_loads_from_mapping():
    cfg = Config(calibration={"meter_host": "shelly.local", "meter_id": 1})
    assert cfg.calibration.meter_host == "shelly.local"
    assert cfg.calibration.meter_id == 1
