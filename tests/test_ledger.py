from tokenwatt.ledger import Ledger, LedgerRow


def _row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000, req_type="text", tok_in=10, cold=False, in_flight=1):
    return LedgerRow(
        ts_start=100.0, ts_end=101.0, model=model,
        e_window_j=marg_j + 100, e_idle_j=100, e_marginal_j=marg_j,
        kwh_marginal=marg_j / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=cost,
        tok_in=tok_in, tok_out=tok_out, tok_source="backend", energy_confidence="estimated (±15-30%)",
        req_type=req_type, cold=cold, in_flight=in_flight,
    )


def test_insert_and_by_model_rollup(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000))
    led.insert(_row(model="m1", marg_j=3_600_000.0, cost=0.31, tok_out=1000))
    rows = led.by_model()
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "m1" and r["requests"] == 2
    assert abs(r["total_usd"] - 0.62) < 1e-9
    assert abs(r["j_per_token"] - (7_200_000.0 / 2000)) < 1e-6


def test_totals_since(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row())
    t = led.totals(since_epoch=0.0)
    assert t["requests"] == 1 and abs(t["usd"] - 0.31) < 1e-9


def test_cost_is_none_when_no_rate(tmp_path):
    # Honesty contract: with no rate, cost must stay NULL through the rollups,
    # never collapse to $0 (which would read as "free").
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=100.0, ts_end=101.0, model="m1",
        e_window_j=10.0, e_idle_j=1.0, e_marginal_j=9.0,
        kwh_marginal=9.0 / 3.6e6, rate_usd_kwh=None, cost_marginal_usd=None,
        tok_in=1, tok_out=1, tok_source="self-count", energy_confidence="estimated (±15-30%)",
    ))
    assert led.by_model()[0]["total_usd"] is None
    assert led.totals(0.0)["usd"] is None


def test_embedding_j_per_token_uses_input(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    # an embedding row: tok_out is None (no output), tok_in carries the embedded tokens.
    led.insert(_row(model="embed", req_type="embedding", tok_out=None, tok_in=500, marg_j=1000.0))
    r = [x for x in led.by_model() if x["model"] == "embed"][0]
    assert r["req_type"] == "embedding"
    assert abs(r["j_per_token"] - (1000.0 / 500)) < 1e-9   # J / INPUT token for embeddings


def test_model_load_summary(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert_model_load(ts=100.0, model="m1", upstream="http://a", load_energy_j=480.0, duration_ms=4800.0, trigger="transition")
    led.insert_model_load(ts=101.0, model="v1", upstream="http://b", load_energy_j=120.0, duration_ms=1200.0, trigger="ttft_outlier")
    s = led.model_load_summary()
    assert s["count"] == 2 and abs(s["total_load_j"] - 600.0) < 1e-9


def test_cold_flag_persists(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", cold=True))
    with led._conn() as c:
        assert c.execute("SELECT cold FROM requests").fetchone()["cold"] == 1


def test_usd_per_mtok(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", cost=0.31, tok_out=1000, marg_j=1.0))   # $0.31 over 1000 out tok
    r = [x for x in led.by_model() if x["model"] == "m1"][0]
    assert abs(r["usd_per_mtok"] - (0.31 / 1000 * 1e6)) < 1e-6           # = $310/Mtok


def test_usd_per_mtok_none_when_no_rate(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", cost=None, tok_out=1000))
    assert [x for x in led.by_model() if x["model"] == "m1"][0]["usd_per_mtok"] is None


def test_in_flight_column_roundtrips(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", in_flight=3))
    with led._conn() as c:
        assert c.execute("SELECT in_flight FROM requests").fetchone()["in_flight"] == 3


def test_in_flight_defaults_to_1(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1"))
    with led._conn() as c:
        assert c.execute("SELECT in_flight FROM requests").fetchone()["in_flight"] == 1


def test_request_id_column_roundtrips(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=100.0, ts_end=101.0, model="m1",
        e_window_j=10.0, e_idle_j=1.0, e_marginal_j=9.0,
        kwh_marginal=9.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.01,
        tok_in=1, tok_out=1, tok_source="backend", energy_confidence="estimated (±15-30%)",
        request_id="abc123",
    ))
    with led._conn() as c:
        assert c.execute("SELECT request_id FROM requests").fetchone()["request_id"] == "abc123"


def test_request_id_defaults_empty(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=100.0, ts_end=101.0, model="m1",
        e_window_j=10.0, e_idle_j=1.0, e_marginal_j=9.0,
        kwh_marginal=9.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.01,
        tok_in=1, tok_out=1, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    with led._conn() as c:
        assert c.execute("SELECT request_id FROM requests").fetchone()["request_id"] == ""
