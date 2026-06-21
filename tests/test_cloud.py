from tokenwatt import cloud


def test_cloud_cost_is_input_plus_output():
    # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
    assert abs(cloud.cloud_cost(1_000_000, 1_000_000, {"in": 0.15, "out": 0.60}) - 0.75) < 1e-9


def test_cheapest_cloud_total_picks_lowest_for_prefill_heavy():
    # prefill-heavy (lots of input): the low-input-price option wins
    table = {"cheap_in": {"in": 0.10, "out": 5.00}, "cheap_out": {"in": 2.00, "out": 0.50}}
    name, usd = cloud.cheapest_cloud_total(100_000, 1_000, table)   # 100:1 input-heavy
    assert name == "cheap_in"
    assert abs(usd - (100_000 * 0.10e-6 + 1_000 * 5.00e-6)) < 1e-12


def test_compare_total_local_cheaper():
    # table-independent: local $0.0092 vs the cheapest entry for 97727 in / 2815 out
    table = {"cheap": {"in": 0.15, "out": 0.60}, "pricey": {"in": 3.0, "out": 15.0}}
    r = cloud.compare_total(0.00918, 97_727, 2_815, table)
    assert r["cloud"] == "cheap"
    assert r["ratio"] > 1.5                       # local meaningfully cheaper on total cost
    assert abs(r["cloud_usd"] - (97_727 * 0.15e-6 + 2_815 * 0.60e-6)) < 1e-9


def test_compare_total_none_when_unpriced_or_no_tokens():
    assert cloud.compare_total(None, 100, 100) is None
    assert cloud.compare_total(0.0, 100, 100) is None
    assert cloud.compare_total(0.01, 0, 0) is None


def test_builtin_table_is_dated_and_has_in_and_out():
    assert cloud.AS_OF and len(cloud.CLOUD_PRICES) >= 2
    for p in cloud.CLOUD_PRICES.values():
        assert "in" in p and "out" in p
