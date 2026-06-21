# TokenWatt M1c-b — The Shareable Cost Story — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ledger into the answer people actually want — `$/Mtok` per model, a `tokenwatt compare` verdict against named cloud prices ("your m1 is N× cheaper than the cheapest cloud"), a copy-pasteable `tokenwatt wrap` card, and a README that leads with the real number — so the tool produces a shareable artifact, not just a database.

**Architecture:** Add a `$/Mtok` column to the existing `by_model()` rollup (cost per million tokens, output tokens for generative/vision, input for embeddings). A tiny `cloud.py` holds a DATED, editable table of representative cloud output prices and a `compare()` that returns the cheapest-cloud ratio. Two new CLI commands consume these: `compare` (per-model local-vs-cloud verdict) and `wrap` (a markdown card + pre-filled share text). The README is rewritten image-first with a real captured report as the hero.

**Tech Stack:** Python 3.12, existing stack (no new deps).

## Global Constraints

- **Python ≥ 3.12.** No new dependencies.
- **`$/Mtok` (cost per million tokens):** `total_usd / denom × 1e6`, where `denom` is `total_out` for generative/vision and `total_in` for embeddings (same denominator as `j_per_token`). `None` when `total_usd` is `None` (no rate) or the denominator is 0 — never a divide-by-zero, never a fake $0.
- **Cloud prices are a DATED, EDITABLE snapshot** (`cloud.py` carries an `AS_OF` string + a docstring telling users to edit to their providers). `compare()` measures against the CHEAPEST cloud entry (the most conservative bar: beating it beats them all). Output prices only.
- **Honesty preserved:** every number stays labeled estimated until calibrated (M2); a model with no rate set shows "no priced tokens (set --rate)", never a fabricated comparison.
- **Existing behavior unchanged:** `report`, `serve`, routing, the per-type ledger, the M1a un-COALESCE'd-cost honesty contract — all untouched except the additive `$/Mtok` column.
- **No sudo at runtime.** Version from `VERSION` (auto-bumped per commit). Conventional Commit prefixes.

---

## File Structure

```
src/tokenwatt/cloud.py     # dated cloud $/Mtok table + cheapest_cloud() + compare()
src/tokenwatt/ledger.py    # MODIFIED: by_model() rows gain usd_per_mtok
src/tokenwatt/cli.py       # MODIFIED: report shows $/Mtok column; new `compare` + `wrap` commands; wrap_card()
README.md                  # MODIFIED: image-first hero with a real captured report
tests/test_cloud.py
tests/test_ledger.py       # MODIFIED: usd_per_mtok assertion
tests/test_report_render.py# MODIFIED: $/Mtok column + wrap_card test
tests/test_cli_compare.py
```

---

### Task 1: `$/Mtok` per model

**Files:**
- Modify: `src/tokenwatt/ledger.py`
- Modify: `tests/test_ledger.py`

**Interfaces:**
- Produces: each `by_model()` row gains `usd_per_mtok = total_usd / denom × 1e6` (`denom` = `total_in` for `embedding`, else `total_out`); `None` when `total_usd` is `None` or `denom` is 0.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_ledger.py`:

```python
def test_usd_per_mtok(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", cost=0.31, tok_out=1000, marg_j=1.0))   # $0.31 over 1000 out tok
    r = [x for x in led.by_model() if x["model"] == "m1"][0]
    assert abs(r["usd_per_mtok"] - (0.31 / 1000 * 1e6)) < 1e-6           # = $310/Mtok


def test_usd_per_mtok_none_when_no_rate(tmp_path):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(_row(model="m1", cost=None, tok_out=1000))
    assert [x for x in led.by_model() if x["model"] == "m1"][0]["usd_per_mtok"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ledger.py -q`
Expected: FAIL — `by_model()` rows have no `usd_per_mtok` key.

- [ ] **Step 3: Modify `by_model` in `ledger.py`**

In the `by_model` loop, after `d["j_per_token"] = ...`, add:

```python
                d["usd_per_mtok"] = (d["total_usd"] / denom * 1e6) if (d["total_usd"] is not None and denom) else None
```

(`denom` is already computed one line above for `j_per_token`; reuse it.)

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_ledger.py -q && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/ledger.py tests/test_ledger.py
git commit -m "feat(ledger): per-model \$/Mtok"
```

---

### Task 2: Cloud price table + comparison

**Files:**
- Create: `src/tokenwatt/cloud.py`
- Test: `tests/test_cloud.py`

**Interfaces:**
- Produces:
  - `AS_OF: str` (e.g. `"2026-06"`); `CLOUD_USD_PER_MTOK: dict[str, float]` — representative cloud OUTPUT $/Mtok.
  - `cheapest_cloud(table=None) -> tuple[str, float]` — the lowest-priced entry.
  - `compare(local_usd_per_mtok: float | None, table=None) -> dict | None` — returns `{"cloud": name, "cloud_usd_per_mtok": price, "ratio": price / local}` (`ratio > 1` ⇒ local cheaper); `None` when local is `None` or `<= 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cloud.py
from tokenwatt import cloud


def test_cheapest_cloud_is_the_min():
    name, price = cloud.cheapest_cloud({"a": 5.0, "b": 1.0, "c": 10.0})
    assert name == "b" and price == 1.0


def test_compare_ratio_local_cheaper():
    # local $0.50/Mtok vs cheapest cloud $1.00 -> 2x cheaper
    r = cloud.compare(0.50, {"x": 1.0, "y": 4.0})
    assert r["cloud"] == "x" and r["cloud_usd_per_mtok"] == 1.0
    assert abs(r["ratio"] - 2.0) < 1e-9


def test_compare_none_when_local_unknown():
    assert cloud.compare(None) is None
    assert cloud.compare(0.0) is None


def test_builtin_table_is_dated_and_nonempty():
    assert cloud.AS_OF and len(cloud.CLOUD_USD_PER_MTOK) >= 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cloud.py -q`
Expected: FAIL — `No module named tokenwatt.cloud`.

- [ ] **Step 3: Implement `cloud.py`**

```python
# src/tokenwatt/cloud.py
from __future__ import annotations

# Representative cloud OUTPUT prices in USD per MILLION tokens. This is a DATED snapshot —
# EDIT it to the providers/models you actually compare against. List prices, output only.
AS_OF = "2026-06"
CLOUD_USD_PER_MTOK: dict[str, float] = {
    "gpt-4o-mini": 0.60,
    "claude-haiku": 1.25,
    "gpt-4o": 10.00,
    "claude-sonnet": 15.00,
}


def cheapest_cloud(table: dict[str, float] | None = None) -> tuple[str, float]:
    """Lowest-priced cloud entry. `table` must be non-empty (an empty price table is a
    programmer error, not a runtime condition to handle)."""
    table = table or CLOUD_USD_PER_MTOK
    name = min(table, key=table.get)
    return name, table[name]


def compare(local_usd_per_mtok: float | None, table: dict[str, float] | None = None) -> dict | None:
    """Compare a local $/Mtok against the CHEAPEST cloud entry (the most conservative bar).
    ratio > 1 means local is that many times cheaper."""
    if local_usd_per_mtok is None or local_usd_per_mtok <= 0:
        return None
    name, price = cheapest_cloud(table)
    return {"cloud": name, "cloud_usd_per_mtok": price, "ratio": price / local_usd_per_mtok}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cloud.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/cloud.py tests/test_cloud.py
git commit -m "feat(cloud): dated editable cloud price table + compare()"
```

---

### Task 3: `tokenwatt compare` + `tokenwatt wrap`

**Files:**
- Modify: `src/tokenwatt/cli.py`
- Test: `tests/test_cli_compare.py`, `tests/test_report_render.py`

**Interfaces:**
- Consumes: `by_model()` rows with `usd_per_mtok` (Task 1); `cloud.compare`/`cheapest_cloud`/`AS_OF` (Task 2); `Ledger.totals`.
- Produces: `compare` CLI command (per-model local-vs-cloud verdict); `wrap_card(ledger, now, days=30) -> str` (markdown card + pre-filled share line) and the `wrap` CLI command. `render_report` gains a `$/Mtok` column.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_compare.py
from typer.testing import CliRunner
from tokenwatt.cli import app, wrap_card
from tokenwatt.ledger import Ledger, LedgerRow

runner = CliRunner()


def _seed(tmp_path, cost=0.0001, tok_out=1000):
    led = Ledger(str(tmp_path / "l.sqlite"))
    led.insert(LedgerRow(
        ts_start=1000.0, ts_end=1001.0, model="m1",
        e_window_j=11.0, e_idle_j=1.0, e_marginal_j=10.0,
        kwh_marginal=10.0 / 3.6e6, rate_usd_kwh=0.31, cost_marginal_usd=cost,
        tok_in=10, tok_out=tok_out, tok_source="backend", energy_confidence="estimated (±15-30%)",
    ))
    return str(tmp_path / "l.sqlite")


def test_compare_command_shows_cheaper_verdict(tmp_path):
    path = _seed(tmp_path)   # $0.0001 / 1000 tok = $0.10/Mtok, way under the cheapest cloud
    res = runner.invoke(app, ["compare", "--ledger", path])
    assert res.exit_code == 0
    assert "m1" in res.stdout and "cheaper" in res.stdout.lower()


def test_wrap_card_has_cost_and_share_text(tmp_path):
    path = _seed(tmp_path)
    card = wrap_card(Ledger(path), now=1002.0, days=30)
    assert "local inference" in card.lower()
    assert "$" in card                      # the electricity figure
    assert "Share:" in card                 # the pre-filled share line
```

Append a `$/Mtok` render assertion to `tests/test_report_render.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_compare.py tests/test_report_render.py -q`
Expected: FAIL — `compare`/`wrap` commands and `wrap_card` don't exist; no `$/Mtok` header.

- [ ] **Step 3: Modify `cli.py`**

**Replace** `render_report`'s table-header line (currently ending `{'J/tok':>10}`) and the per-model loop with the version below — do NOT add a second header or loop:

```python
        f"  {'model':<16}{'type':>11}{'req':>6}{'kWh':>12}{'$':>10}{'J/tok':>10}{'$/Mtok':>10}",
    ]
    for r in ledger.by_model():
        jpt = f"{r['j_per_token']:.3f}" if r["j_per_token"] is not None else "-"
        pm = _pmtok(r["usd_per_mtok"])
        lines.append(
            f"  {r['model']:<16}{r['req_type']:>11}{r['requests']:>6}{r['total_kwh']:>12.4f}"
            f"{_usd(r['total_usd']):>10}{jpt:>10}{pm:>10}"
        )
    return "\n".join(lines)
```

Add `wrap_card` and the two commands (near the other commands):

```python
from tokenwatt import cloud


def _pmtok(v: float | None) -> str:
    """$/Mtok display (define near _usd): '-' when unknown, '<0.001' for a tiny positive —
    never collapse a real positive value to a fake '0.000'."""
    if v is None:
        return "-"
    return "<0.001" if 0 < v < 0.001 else f"{v:.3f}"


def wrap_card(ledger: Ledger, now: float, days: int = 30) -> str:
    t = ledger.totals(now - days * 86_400)
    lines = [
        f"## My local inference — last {days} days",
        f"- {t['requests']} requests · {t['kwh']:.3f} kWh · {_usd(t['usd'])} in electricity",
    ]
    best = None   # the model with the lowest $/Mtok that has a price
    for r in ledger.by_model():
        if r["usd_per_mtok"] is not None:
            lines.append(f"- {r['model']} ({r['req_type']}): ${_pmtok(r['usd_per_mtok'])}/Mtok electricity")
            if best is None or r["usd_per_mtok"] < best:
                best = r["usd_per_mtok"]
    share = "I metered my local LLM electricity with TokenWatt."
    if best is not None:
        c = cloud.compare(best)
        if c and c["ratio"] >= 1:
            lines.append(f"- vs cloud: ~{c['ratio']:.0f}× cheaper than {c['cloud']} (${c['cloud_usd_per_mtok']:.2f}/Mtok output)")
            share = (f"I ran inference locally for {days} days — {_usd(t['usd'])} in electricity, "
                     f"~{c['ratio']:.0f}× cheaper per token than {c['cloud']} "
                     f"(cloud prices {cloud.AS_OF}; estimated, pre-calibration). via TokenWatt")
    lines += ["", f"Share: {share}"]
    return "\n".join(lines)


@app.command()
def compare(ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger")):
    """Compare your per-model electricity $/Mtok against named cloud prices."""
    import os
    led = Ledger(os.path.expanduser(ledger))
    typer.echo(f"local electricity vs cloud output $/Mtok (cloud snapshot {cloud.AS_OF}; edit cloud.py):")
    for r in led.by_model():
        lpm = r["usd_per_mtok"]
        if lpm is None:
            typer.echo(f"  {r['model']:<20} no priced tokens (set --rate)")
            continue
        c = cloud.compare(lpm)
        if c is None:                      # priced (e.g. $0 at --rate 0) but no cloud verdict
            typer.echo(f"  {r['model']:<20} ${_pmtok(lpm)}/Mtok electricity (no cloud comparison)")
            continue
        verdict = (f"{c['ratio']:.1f}× cheaper than {c['cloud']} (${c['cloud_usd_per_mtok']:.2f}/Mtok)"
                   if c["ratio"] >= 1 else
                   f"{1 / c['ratio']:.1f}× MORE than {c['cloud']}")
        typer.echo(f"  {r['model']:<20} ${_pmtok(lpm)}/Mtok electricity — {verdict}")


@app.command()
def wrap(ledger: str = typer.Option("~/.tokenwatt/ledger.sqlite", "--ledger"),
         days: int = typer.Option(30, "--days")):
    """Print a shareable markdown card of your local-inference cost."""
    import os
    typer.echo(wrap_card(Ledger(os.path.expanduser(ledger)), now=time.time(), days=days))
```

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_cli_compare.py tests/test_report_render.py -q && uv run pytest -q`
Expected: PASS (all; the existing render tests still pass — the new column is additive and the existing assertions check `m1`/`$0.31`/`estimated`/`—`).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/cli.py tests/test_cli_compare.py tests/test_report_render.py
git commit -m "feat(cli): \$/Mtok column + compare + wrap shareable card"
```

---

### Task 4: README hero

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (docs). Produces the project's front page.

- [ ] **Step 1: Capture a real report block (on-device, hermetic enough)**

```bash
cd ~/mlx-cost-project
uv run tokenwatt report --ledger /tmp/tw-m1b.sqlite 2>/dev/null | sed 's/^/    /'   # if a ledger exists; else use any test ledger
```

Use that captured block as the README hero. (If no ledger is handy, run a quick request through `serve` first, or paste the `compare` output instead.)

- [ ] **Step 2: Rewrite `README.md` image-first**

Replace `README.md` with (fill the captured report block into the fenced section):

```markdown
# TokenWatt

**Know what your local LLM inference actually costs you in electricity — per model, per request, on Apple Silicon. No sudo.**

```text
<PASTE the captured `tokenwatt report` block here>
```

## Install

```bash
uv tool install tokenwatt          # or: uvx tokenwatt
```

Try it without installing:
`uvx --from git+https://github.com/mmmugh/tokenwatt@v0.1.x tokenwatt serve`

## Use

```bash
tokenwatt serve -c tokenwatt.yaml      # one port in front of your local backends; no API key, no sudo
# point your OpenAI client (or Pi / OpenClaw / Claude Code) at http://127.0.0.1:7000
tokenwatt report                       # today/month $, per-model $/Mtok and J/token
tokenwatt compare                      # your $/Mtok vs named cloud prices
tokenwatt wrap                         # a shareable "my inference bill" card
```

`tokenwatt init` scaffolds a commented `tokenwatt.yaml` (routes m1/v1/embeddings, your `$/kWh` or a time-of-use schedule).

## What it does

A transparent, OpenAI-compatible proxy: it forwards each request byte-exact to your local server (`mlx-openai-server`, `mlx-vlm`, Ollama, LM Studio, llama.cpp…), measures the real per-rail SoC energy via Apple's IOReport (no sudo), subtracts an idle baseline, books model-load energy separately, prices it at your utility rate, and logs a per-request, per-model ledger. Numbers are labeled **estimated** until you calibrate against a wall meter.

> Often the honest answer on Apple Silicon is "cheaper than you think." TokenWatt tells you which — for *your* machine and *your* rate.

See the design spec in `docs/design/specs/`.
```

(An animated GIF of the live cost ticking is a nice future hero; a static captured report block is the v1.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): image-first hero with a real report + compare/wrap"
```

---

## Self-Review

**1. Spec coverage (M1c-b slice — spec §17 adoption / shareable hook + §11 reporting):**
- `$/Mtok` per model → Task 1 + Task 3 (report column). ✓
- Cloud break-even (`compare`) → Task 2 (`cloud.py`) + Task 3 (`compare` command). ✓
- Shareable `wrap` card (markdown + share text; PNG deferred) → Task 3 (`wrap_card` + `wrap`). ✓
- README hero (image-first; animated GIF deferred) → Task 4. ✓
- Honesty: `$/Mtok` is `None` (rendered `-`) when no rate; cloud table dated + editable; compare refuses a verdict without a local price → Tasks 1/2/3. ✓
- Deferred correctly (NOT M1c-b): PNG/GIF generation; live cloud-price fetching (the table is a static editable snapshot); carbon/gCO2e; M2 calibration.

**2. Placeholder scan:** no TBD/TODO; every code step complete; commands have expected output. The README captured-block step is an explicit on-device capture, not a placeholder.

**3. Type consistency:** `by_model()` row gains `usd_per_mtok` (Task 1) consumed in `render_report`/`wrap_card`/`compare` (Task 3). `cloud.compare(local) -> {cloud, cloud_usd_per_mtok, ratio} | None` and `cheapest_cloud() -> (name, price)` used identically in Tasks 2/3. `wrap_card(ledger, now, days=30) -> str` matches its CLI caller. The existing render tests' assertions (`m1`/`$0.31`/`estimated`/`—`) survive the additive `$/Mtok` column. ✓

**Known limitation (stated):** cloud prices are a static, dated, hand-edited snapshot — not fetched live; `compare`/`wrap` verdicts are only as current as `cloud.py`. The shareable artifact is markdown/text (no rendered PNG/GIF in v1).
