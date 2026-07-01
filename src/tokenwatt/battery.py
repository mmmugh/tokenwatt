from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadCell:
    """One chat load cell: a standardized prompt + generation cap that stresses
    the SoC a particular way. Shipped in-package so the battery is identical
    across machines and runs (version-stamped into the C2 profile)."""
    name: str
    prompt: str
    max_tokens: int


# A neutral paragraph repeated to a few thousand tokens of context. Deterministic
# and content-free on purpose — we are exercising the machine, not the model.
_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the calibration harness "
    "measures the electricity it takes to think about foxes and dogs and energy. "
)


def _prefill_prompt() -> str:
    # ≈4k tokens of context -> compute-bound prefill (kept modest to fit small
    # context windows; the point is a big prompt, not the largest possible one)
    return (_PARAGRAPH * 120) + "\nReply with a single word: acknowledged."


def _decode_prompt() -> str:
    # tiny prompt, ask for a long continuation -> memory-bound decode
    return "Write an extremely long, detailed, continuous story. Do not stop early."


def _balanced_prompt() -> str:
    return (_PARAGRAPH * 12) + "\nSummarize the passage in a few sentences."


def text_cells() -> list[LoadCell]:
    """The Core-4 battery's three LOAD cells (idle is measured separately)."""
    return [
        LoadCell("prefill", _prefill_prompt(), max_tokens=8),
        LoadCell("decode", _decode_prompt(), max_tokens=1024),
        LoadCell("balanced", _balanced_prompt(), max_tokens=256),
    ]
