from __future__ import annotations

# Representative cloud LIST prices in USD per MILLION tokens, as (input, output). This is a
# DATED snapshot — EDIT it to the providers/models you actually compare against. A cloud LIST
# price includes the provider's compute AND margin, not just electricity.
AS_OF = "2026-06"
CLOUD_PRICES: dict[str, dict[str, float]] = {
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gpt-5-mini":            {"in": 0.25, "out": 2.00},
    "claude-haiku":          {"in": 1.00, "out": 5.00},
    "gpt-5":                 {"in": 1.25, "out": 10.00},
    "claude-sonnet":         {"in": 3.00, "out": 15.00},
}


def cloud_cost(tok_in: int, tok_out: int, prices: dict[str, float]) -> float:
    """USD a cloud model with these (in/out) $/Mtok prices would charge for this token volume."""
    return tok_in / 1e6 * prices["in"] + tok_out / 1e6 * prices["out"]


def cheapest_cloud_total(tok_in: int, tok_out: int, table: dict | None = None) -> tuple[str, float]:
    """(name, usd) of the cheapest cloud option for this exact token volume (input+output)."""
    table = table or CLOUD_PRICES
    name = min(table, key=lambda n: cloud_cost(tok_in, tok_out, table[n]))
    return name, cloud_cost(tok_in, tok_out, table[name])


def compare_total(local_usd: float | None, tok_in: int, tok_out: int,
                  table: dict | None = None) -> dict | None:
    """Compare measured local electricity against the CHEAPEST cloud TOTAL (input+output) for the
    same token volume. Returns {"cloud", "cloud_usd", "ratio"} where ratio = cloud/local
    (>1 => local cheaper). None when local is unpriced/zero or there are no tokens."""
    if local_usd is None or local_usd <= 0 or (tok_in + tok_out) <= 0:
        return None
    name, cloud_usd = cheapest_cloud_total(tok_in, tok_out, table)
    return {"cloud": name, "cloud_usd": cloud_usd, "ratio": cloud_usd / local_usd}
