from tokenwatt.meter import EnergyByRail
from tokenwatt.rate import FlatRate
from tokenwatt.cost import marginal_kwh


def test_marginal_kwh_subtracts_idle():
    window = EnergyByRail({"gpu": 3_600_000.0})   # 3.6 MJ = 1 kWh
    idle = EnergyByRail({"gpu": 1_800_000.0})     # 0.5 kWh
    assert marginal_kwh(window, idle) == 0.5


def test_flat_rate_prices_and_handles_unset():
    assert FlatRate(0.31).price(0.5) == 0.155
    assert FlatRate(None).price(0.5) is None
