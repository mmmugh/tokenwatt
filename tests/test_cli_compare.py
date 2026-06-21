from typer.testing import CliRunner
from tokenwatt.cli import app, wrap_card
from tokenwatt.ledger import Ledger, LedgerRow

runner = CliRunner()


def test_compare_command_total_cost_cheaper_verdict(tmp_path):
    # seed a prefill-heavy, cheaply-metered model: big input, tiny electricity -> local wins
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="qwen3.6-27b",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0001,
        tok_in=100_000, tok_out=1_000, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    res = runner.invoke(app, ["compare", "--ledger", str(tmp_path / "l.sqlite")])
    assert res.exit_code == 0
    assert "qwen3.6-27b" in res.stdout and "cheaper" in res.stdout.lower()
    assert "for the SAME tokens" in res.stdout          # the honest framing is present


def test_wrap_card_total_cost_and_caveat(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="qwen3.6-27b",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0001,
        tok_in=100_000, tok_out=1_000, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    card = wrap_card(led, now=1002.0, days=30)
    assert "input + " in card                            # token-volume line present
    assert "±15-30%" in card                             # UNCONDITIONAL caveat in the body
    assert "Share:" in card


def _seed_pricey(tmp_path):
    # expensive electricity for very few tokens -> local is far PRICIER than the cheapest cloud
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.50,
        tok_in=1_000, tok_out=1_000, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    return led


def test_wrap_card_pricier_not_mislabeled_comparable(tmp_path):
    # THE blocker the final review caught: a pricier-than-cloud result must NOT be called
    # 'comparable within measurement uncertainty' on the shareable card.
    card = wrap_card(_seed_pricey(tmp_path), now=1002.0, days=30)
    assert "pricier" in card.lower()
    assert "comparable" not in card.lower()
    assert "cheaper" not in card.lower()


def test_compare_command_pricier_verdict(tmp_path):
    _seed_pricey(tmp_path)
    res = runner.invoke(app, ["compare", "--ledger", str(tmp_path / "l.sqlite")])
    assert res.exit_code == 0
    assert "pricier" in res.stdout.lower() and "comparable" not in res.stdout.lower()


def test_wrap_card_comparable_middle_band(tmp_path):
    # local ≈ cheapest-cloud total (ratio in the 1/1.5..1.5 band) -> 'comparable', not a winner
    led = Ledger(str(tmp_path / "l.sqlite"))
    # cheapest cloud (gemini-2.5-flash-lite) for 1000 in / 1000 out = 1000*0.10e-6 + 1000*0.40e-6 = $0.0005
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.0005,
        tok_in=1_000, tok_out=1_000, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    card = wrap_card(led, now=1002.0, days=30)
    assert "comparable" in card.lower()
    assert "cheaper" not in card.lower() and "pricier" not in card.lower()


def test_wrap_card_all_unpriced_no_crash(tmp_path):
    # no rate set -> total_usd None: card must not crash, must show '—' not $0, no cloud line
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=None, cost_marginal_usd=None,
        tok_in=100, tok_out=100, tok_source="backend", energy_confidence="energy-only",
    ))
    card = wrap_card(led, now=1002.0, days=30)
    assert "—" in card                                   # unpriced renders as em dash, never $0
    assert "vs cloud" not in card                        # no comparison when unpriced
    assert "I metered my local LLM electricity with TokenWatt." in card   # generic share


def test_compare_zero_tokens_message_not_set_rate(tmp_path):
    # priced energy-only row with 0 tokens: don't misinstruct 'set --rate' (rate IS set)
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=0.001,
        tok_in=0, tok_out=0, tok_source="none", energy_confidence="energy-only",
    ))
    res = runner.invoke(app, ["compare", "--ledger", str(tmp_path / "l.sqlite")])
    assert res.exit_code == 0
    assert "no tokens to compare" in res.stdout and "set --rate" not in res.stdout
