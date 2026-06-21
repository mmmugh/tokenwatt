from tokenwatt.meter import EnergyByRail, FakeMeter, RAILS


def test_energybyrail_total_and_sub():
    a = EnergyByRail({"gpu": 5.0, "dram": 3.0})
    b = EnergyByRail({"gpu": 2.0, "dram": 4.0})
    assert a.total_j == 8.0
    diff = a - b
    assert diff.joules["gpu"] == 3.0
    assert diff.joules["dram"] == 0.0          # clamped at zero, never negative


def test_from_zeus_converts_mj_to_j_and_skips_none():
    class M:  # mimics AppleEnergyMetrics
        cpu_total_mj = 1000
        gpu_mj = 2000
        gpu_sram_mj = None
        dram_mj = 500
        ane_mj = None
    e = EnergyByRail.from_zeus(M())
    assert e.joules == {"cpu_total": 1.0, "gpu": 2.0, "dram": 0.5}
    assert e.total_j == 3.5


def test_fake_meter_window_default_one_joule_per_rail():
    m = FakeMeter()
    m.begin("r1")
    w = m.end("r1")
    assert w.total_j == float(len(RAILS))
