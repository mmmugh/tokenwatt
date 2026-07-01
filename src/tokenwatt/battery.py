from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


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


@dataclass(frozen=True)
class ChatResult:
    tok_in: int | None      # None when the upstream reports no usage — never a fabricated 0
    tok_out: int | None


@runtime_checkable
class LoadClient(Protocol):
    def chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult: ...


class HttpLoadClient:
    """Drives inference load by POSTing OpenAI chat completions to an upstream.
    Non-streaming so token usage is read straight from the response body."""

    def __init__(self, upstream: str, client: httpx.Client | None = None, timeout: float = 120.0) -> None:
        self._upstream = upstream.rstrip("/")
        self._timeout = timeout
        self._own = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    def chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult:
        r = self._client.post(
            f"{self._upstream}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "stream": False, "temperature": 0.0},
            timeout=self._timeout,
        )
        r.raise_for_status()
        usage = r.json().get("usage")            # None if the key is missing OR explicitly null
        if not usage:                            # None / null / {} -> unknown, NOT a fabricated 0
            return ChatResult(tok_in=None, tok_out=None)
        pin, pout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        return ChatResult(tok_in=int(pin) if pin is not None else None,
                          tok_out=int(pout) if pout is not None else None)

    def close(self) -> None:
        if self._own:
            self._client.close()


class FakeLoadClient:
    """Deterministic test double: fixed token counts, and (given a fake clock)
    advances it by `latency_s` per call so a wall-clock load loop terminates."""

    def __init__(self, tok_in: int = 100, tok_out: int = 50,
                 latency_s: float = 1.0, clock=None) -> None:
        self._r = ChatResult(tok_in=tok_in, tok_out=tok_out)
        self._latency = latency_s
        self._clock = clock

    def chat(self, model: str, prompt: str, max_tokens: int) -> ChatResult:
        if self._clock is not None:
            self._clock.sleep(self._latency)
        return self._r
