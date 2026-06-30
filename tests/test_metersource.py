import pytest

from tokenwatt.metersource import MeterSource, FakeMeterSource


def test_fake_meter_source_returns_scripted_readings_in_order():
    src = FakeMeterSource([100.0, 100.001, 100.5])
    assert src.read_accumulated_wh() == 100.0
    assert src.read_accumulated_wh() == 100.001
    assert src.read_accumulated_wh() == 100.5


def test_fake_meter_source_holds_last_reading_when_exhausted():
    src = FakeMeterSource([7.0])
    assert src.read_accumulated_wh() == 7.0
    assert src.read_accumulated_wh() == 7.0  # does not raise; sustained-reads safe


def test_fake_meter_source_satisfies_protocol():
    # the protocol is the contract the harness depends on; a source missing
    # name/tier/accuracy or read_accumulated_wh must NOT type as a MeterSource
    assert isinstance(FakeMeterSource([1.0]), MeterSource)


def test_fake_meter_source_rejects_empty_readings():
    with pytest.raises(ValueError):
        FakeMeterSource([])


import httpx

from tokenwatt.metersource import ShellyMeterSource, ShellyStatus, MeterSource


_SHELLY_BODY = {       # shape of GET /rpc/Switch.GetStatus?id=0 (Gen2, unwrapped)
    "id": 0, "output": True, "apower": 12.4, "voltage": 121.7, "current": 0.10,
    "aenergy": {"total": 1234.567, "by_minute": [0, 0, 0], "minute_ts": 1700000000},
}


def _mock_shelly(body=_SHELLY_BODY, status=200):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, seen


def test_shelly_reads_accumulated_wh_from_aenergy_total():
    client, seen = _mock_shelly()
    src = ShellyMeterSource("shelly.local", switch_id=0, client=client)
    assert src.read_accumulated_wh() == pytest.approx(1234.567)
    # hits the Gen2 RPC GET endpoint with the right component id
    assert seen["url"] == "http://shelly.local/rpc/Switch.GetStatus?id=0"


def test_shelly_read_status_parses_power_and_voltage():
    client, _ = _mock_shelly()
    src = ShellyMeterSource("shelly.local", client=client)
    s = src.read_status()
    assert isinstance(s, ShellyStatus)
    assert (s.total_wh, s.apower_w, s.voltage_v) == pytest.approx((1234.567, 12.4, 121.7))


def test_shelly_satisfies_meter_source_protocol():
    client, _ = _mock_shelly()
    assert isinstance(ShellyMeterSource("h", client=client), MeterSource)


def test_shelly_reachable_true_summarizes_status():
    client, _ = _mock_shelly()
    ok, detail = ShellyMeterSource("h", client=client).reachable()
    assert ok is True
    assert "1234.567" in detail and "12.4" in detail


def test_shelly_reachable_false_on_http_error():
    client, _ = _mock_shelly(status=500)
    ok, detail = ShellyMeterSource("h", client=client).reachable()
    assert ok is False
    assert "500" in detail or "Error" in detail
