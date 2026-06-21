from tokenwatt.ledger import Ledger, LedgerRow
from tokenwatt.cli import render_report


def test_render_report_shows_model_and_dollars(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=3_600_100.0, e_idle_j=100.0, e_marginal_j=3_600_000.0,
        kwh_marginal=1.0, rate_usd_kwh=0.31, cost_marginal_usd=0.31,
        tok_in=10, tok_out=1000, tok_source="backend", energy_confidence="estimated",
    ))
    text = render_report(led, now=1002.0)
    assert "m1" in text
    assert "$0.31" in text
    assert "estimated" in text.lower()   # honesty banner


def test_render_report_dashes_when_no_rate(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=10.0, e_idle_j=1.0, e_marginal_j=9.0,
        kwh_marginal=9.0 / 3.6e6, rate_usd_kwh=None, cost_marginal_usd=None,
        tok_in=1, tok_out=1, tok_source="self-count", energy_confidence="estimated (±15-30%)",
    ))
    text = render_report(led, now=1002.0)
    assert "$0.0000" not in text   # never render an unpriced row as free
    assert "—" in text


def test_render_report_shows_type_and_embedding_j_per_token(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="embed",
        e_window_j=1001.0, e_idle_j=1.0, e_marginal_j=1000.0,
        kwh_marginal=1000.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0001,
        tok_in=500, tok_out=None, tok_source="backend", energy_confidence="estimated (±15-30%)",
        req_type="embedding",
    ))
    text = render_report(led, now=1002.0)
    embed_line = next(l for l in text.splitlines() if "embed" in l)
    assert "embedding" in embed_line    # the type column is shown for this row
    assert "2.000" in embed_line        # J/tok = 1000 / 500 INPUT tokens; renders '-' if denom wrongly used tok_out=None
    assert "J/tok" in text              # column header present


def test_render_report_shows_model_load_summary(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert_model_load(ts=1000.0, model="m1", upstream="http://a", load_energy_j=480.0, duration_ms=4800.0, trigger="transition")
    text = render_report(led, now=1002.0)
    assert "model loads: 1" in text        # the summary line with the right count


def test_render_report_has_usd_per_mtok_column(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0001,
        tok_in=10, tok_out=1000, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    text = render_report(led, now=1002.0)
    assert "$/Mtok" in text          # the column header
    assert "0.100" in text           # the per-ROW value: $0.0001 / 1000 out-tok × 1e6 = $0.100/Mtok
                                     # (fails if the loop body forgot to append the per-model column)


def test_usd_sub_floor_positive_not_fake_zero():
    from tokenwatt.cli import _usd
    assert _usd(0.00003) == "<$0.0001"   # real, below the 4-dp floor — not "$0.0000"
    assert _usd(0.0) == "$0.0000"        # a genuine zero stays zero (e.g. --rate 0)
    assert _usd(0.0123) == "$0.0123"     # normal value unchanged
    assert _usd(None) == "—"             # no rate -> em dash (unchanged honesty contract)


def test_kwh_sub_floor_positive_not_fake_zero():
    from tokenwatt.cli import _kwh
    assert _kwh(0.00003) == "<0.0001"
    assert _kwh(0.0) == "0.0000"
    assert _kwh(0.5) == "0.5000"


def test_render_shortens_namespaced_model_label(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="mlx-community/Qwen3-VL-8B-Instruct-8bit",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0002,
        tok_in=10, tok_out=1000, tok_source="backend", energy_confidence="estimated (±15-30%)",
        req_type="vision",
    ))
    text = render_report(led, now=1002.0)
    assert "Qwen3-VL-8B-Instruct-8bit" in text          # namespace stripped for display
    assert "mlx-community/" not in text                 # the long prefix is gone


def test_render_headline_sub_floor_cost_is_honest(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.00003,
        tok_in=10, tok_out=1000, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    text = render_report(led, now=1002.0)
    assert "<$0.0001" in text          # the real-but-tiny cost is shown, not "$0.0000"
    assert "$0.0000" not in text       # no fake zero anywhere
