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
