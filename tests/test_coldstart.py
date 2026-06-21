from tokenwatt.coldstart import ColdStartDetector


def test_first_observation_seeds_baseline_and_is_not_cold():
    # First sight of a model has no warm baseline -> it SEEDS one and is NOT flagged cold
    # (avoids permanently mis-flagging a naturally-slow warm model).
    d = ColdStartDetector()
    r = d.observe("m1", ttft_s=5.0, window_marginal_j=600.0, duration_s=6.0)
    assert r.is_cold is False and r.load_energy_j == 0.0


def test_warm_request_is_not_cold():
    d = ColdStartDetector()
    assert d.observe("m1", 0.3, 100.0, 2.0).is_cold is False


def test_cold_books_energy_by_time_fraction():
    # Warm baseline ~0.2s, then a 5s TTFT on a 6s request -> load (5-0.2)/6 of 600 J = 480 J.
    d = ColdStartDetector(floor_s=0.5)               # baseline*3 (0.6) governs over the floor
    for _ in range(3):
        d.observe("m1", 0.2, 100.0, 2.0)             # baseline median 0.2
    r = d.observe("m1", ttft_s=5.0, window_marginal_j=600.0, duration_s=6.0)
    assert r.is_cold is True
    assert abs(r.load_energy_j - (4.8 / 6.0) * 600.0) < 1e-6   # 480 J
    assert abs(r.load_time_s - 4.8) < 1e-9
    assert r.trigger == "ttft_outlier"               # same model -> not a transition


def test_reload_uses_baseline_threshold_not_just_floor():
    # warm baseline 0.5 -> threshold = max(floor 1.0, 0.5*3=1.5) = 1.5, so the BASELINE governs.
    d = ColdStartDetector()
    for _ in range(4):
        d.observe("m1", 0.5, 100.0, 2.0)
    assert d.observe("m1", 1.2, 100.0, 2.0).is_cold is False   # below 1.5 -> warm (would be cold under floor-only)
    assert d.observe("m1", 4.0, 500.0, 5.0).is_cold is True    # above 1.5 -> cold


def test_non_streaming_none_ttft_is_never_cold():
    d = ColdStartDetector()
    r = d.observe("m1", ttft_s=None, window_marginal_j=999.0, duration_s=10.0)
    assert r.is_cold is False and r.load_energy_j == 0.0
