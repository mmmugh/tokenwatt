from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass
class ColdResult:
    is_cold: bool
    load_energy_j: float   # TIME-fraction ESTIMATE: assumes uniform power across the window, so
                           # it OVER-books when the load phase draws less power than decode.
                           # An estimate, not a measured load energy.
    load_time_s: float     # estimated load duration (ttft - warm baseline)
    trigger: str           # "transition" | "ttft_outlier" | "none"


class ColdStartDetector:
    """Detects model-load (cold-start) events from time-to-first-token and estimates the
    load energy. A load takes seconds; warm prefill is sub-second.

    The FIRST observation of a model SEEDS its warm baseline and is never flagged cold —
    this misses a genuine first-ever load (booked as inference) but is far safer than
    permanently mis-flagging a model whose warm TTFT naturally exceeds floor_s. Cold TTFTs
    do NOT feed the baseline."""

    def __init__(self, window: int = 16, floor_s: float = 1.0, factor: float = 3.0) -> None:
        self._last_model: str | None = None
        self._ttft: dict[str, deque[float]] = {}
        self._window = window
        self._floor_s = floor_s
        self._factor = factor

    def observe(self, model: str, ttft_s: float | None,
                window_marginal_j: float, duration_s: float) -> ColdResult:
        # observe() MUST stay synchronous and single-threaded: the proxy calls it from the
        # event loop with no await between the read and write of self._ttft / self._last_model
        # (same discipline as IdleBaseline). An await here, or moving finalization to a thread
        # pool, would corrupt the baseline deque.
        # `trigger` is ADVISORY only (a cosmetic tag on the model_loads row). The cold/warm
        # decision is purely TTFT-vs-baseline and does NOT depend on it; under concurrent
        # requests _last_model reflects finalize order, so "transition" is best-effort.
        transition = model != self._last_model
        self._last_model = model

        if ttft_s is None:                       # non-streaming: can't measure a split
            return ColdResult(False, 0.0, 0.0, "none")

        samples = self._ttft.get(model)
        baseline = median(samples) if samples else None
        if baseline is None:                     # no warm baseline yet: SEED it, never cold
            self._ttft.setdefault(model, deque(maxlen=self._window)).append(ttft_s)
            return ColdResult(False, 0.0, 0.0, "none")

        threshold = max(self._floor_s, baseline * self._factor)
        if ttft_s <= threshold:                  # warm: feed the baseline, not cold
            self._ttft[model].append(ttft_s)
            return ColdResult(False, 0.0, 0.0, "none")

        load_time = max(0.0, ttft_s - baseline)
        frac = min(1.0, load_time / duration_s) if duration_s > 0 else 0.0
        return ColdResult(True, frac * window_marginal_j, load_time,
                          "transition" if transition else "ttft_outlier")
