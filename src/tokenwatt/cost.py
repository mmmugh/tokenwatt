from __future__ import annotations
from tokenwatt.meter import EnergyByRail


def marginal_kwh(window: EnergyByRail, idle: EnergyByRail) -> float:
    """Marginal energy (window minus idle baseline) in kWh."""
    return (window - idle).total_j / 3.6e6
