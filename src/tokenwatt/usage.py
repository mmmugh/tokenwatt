from __future__ import annotations
import json
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")  # model-agnostic approximation


@dataclass
class TokenUsage:
    input: int | None
    output: int | None
    cached: int | None
    source: str       # "backend" | "self-count" | "none"
    confidence: str   # "high" | "low" | "energy-only"


def _backend_usage(usage: dict) -> TokenUsage:
    """Build a TokenUsage from an OpenAI `usage` object (backend-authoritative)."""
    details = usage.get("prompt_tokens_details") or {}
    return TokenUsage(
        input=usage.get("prompt_tokens"),
        output=usage.get("completion_tokens"),
        cached=details.get("cached_tokens"),
        source="backend",
        confidence="high",
    )


def usage_from_response_json(body: dict) -> TokenUsage | None:
    usage = body.get("usage")
    if not usage:
        return None
    return _backend_usage(usage)


def _count(text: str) -> int:
    return len(_ENC.encode(text or ""))


class SelfCounter:
    """Accounts for a streamed response. Prefers the backend's in-stream `usage` chunk
    (sent when stream_options.include_usage is set); falls back to self-counting the
    `content` + `reasoning_content` deltas with tiktoken when no usage chunk arrives."""

    def __init__(self, request_body: dict) -> None:
        msgs = request_body.get("messages") or []
        self._input = sum(
            _count(m.get("content", "")) for m in msgs if isinstance(m.get("content"), str)
        )
        self._buf = ""
        self._out_text = ""
        self._backend: dict | None = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk.decode("utf-8", errors="ignore")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                self._backend = usage   # backend reported real counts in-stream
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                for field in ("content", "reasoning_content"):
                    if isinstance(delta.get(field), str):
                        self._out_text += delta[field]
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    for k in ("name", "arguments"):
                        if isinstance(fn.get(k), str):
                            self._out_text += fn[k]

    def result(self) -> TokenUsage:
        if self._backend is not None:
            return _backend_usage(self._backend)
        return TokenUsage(
            input=self._input,
            output=_count(self._out_text),
            cached=None,
            source="self-count",
            confidence="low",
        )
