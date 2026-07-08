import pytest

from tokenwatt.powersource import (
    BatteryFlux, BatterySource, IOKitBatterySource, FakeBatterySource, _parse,
)

_CHARGING = '  "InstantAmperage" = 1500\n  "Voltage" = 12000\n  "IsCharging" = Yes\n'
# 2**64 - 1200 == InstantAmperage reported for -1200 mA (two's-complement wrap on discharge)
_DISCHARGING = '  "InstantAmperage" = 18446744073709550416\n  "Voltage" = 11400\n  "IsCharging" = No\n'
_NO_BATTERY = '  "InstantAmperage" = 0\n  "Voltage" = 0\n  "IsCharging" = No\n'


def test_parse_charging_uses_magnitude_and_flag():
    f = _parse(_CHARGING)
    assert f == BatteryFlux(abs_w=pytest.approx(18.0), charging=True)   # 1.5 A * 12 V


def test_parse_discharging_is_positive_magnitude_flag_false():
    f = _parse(_DISCHARGING)
    # magnitude only: |-1.2 A| * 11.4 V = 13.68 W; sign never enters the gate
    assert f.abs_w == pytest.approx(13.68) and f.charging is False


def test_parse_returns_none_when_no_real_battery():
    assert _parse(_NO_BATTERY) is None            # Voltage == 0 -> desktop


def test_parse_returns_none_on_missing_fields():
    assert _parse("nothing useful here") is None


# Real battery ioreg (captured from an M3 MacBook Air) packs fields comma-separated with
# NO spaces and inlines a big LifetimeData dict right after Voltage. A desktop stub / clean
# fixtures hid this; the value regex must stop at the comma, not run to the next whitespace.
_REAL_BATTERY = (
    '  "Amperage" = 0\n'
    '  "ExternalConnected" = Yes\n'
    '  "Voltage" = 12590,"LifetimeData"={"Raw"=<02f1f13c00005eb3>,"UpdateTime"=1783483379}\n'
    '  "InstantAmperage" = 1000,"PostChargeWaitSeconds"=120\n'
    '  "IsCharging" = Yes\n'
)


def test_parse_handles_real_comma_packed_ioreg_fields():
    # regression: value must parse as just the number, not "12590,\"LifetimeData\"=..."
    f = _parse(_REAL_BATTERY)
    assert f == BatteryFlux(abs_w=pytest.approx(1.0 * 12.59), charging=True)  # 1.0 A * 12.59 V


def test_iokit_source_parses_injected_ioreg_output():
    def fake_run(args, text=True):
        assert args == ["ioreg", "-rn", "AppleSmartBattery"]
        return _CHARGING
    assert IOKitBatterySource(run=fake_run).read_flux() == BatteryFlux(18.0, True)


def test_iokit_source_returns_none_when_ioreg_absent():
    def boom(*a, **k):
        raise FileNotFoundError("ioreg")          # non-Mac CI / no such binary
    assert IOKitBatterySource(run=boom).read_flux() is None


def test_fake_source_replays_then_holds_last_and_satisfies_protocol():
    src = FakeBatterySource([BatteryFlux(20.0, True), None])
    assert isinstance(src, BatterySource)
    assert src.read_flux() == BatteryFlux(20.0, True)
    assert src.read_flux() is None
    assert src.read_flux() is None                # holds last; sustained reads never raise


def test_fake_source_rejects_empty():
    with pytest.raises(ValueError):
        FakeBatterySource([])
