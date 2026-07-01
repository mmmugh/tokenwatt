import httpx
import pytest

from tokenwatt.battery import ChatResult, LoadCell, LoadClient, HttpLoadClient, FakeLoadClient, text_cells


def test_text_cells_are_the_core_three_load_cells():
    cells = text_cells()
    assert [c.name for c in cells] == ["prefill", "decode", "balanced"]
    assert all(isinstance(c, LoadCell) for c in cells)


def test_prefill_is_long_prompt_short_generation():
    # prefill must be compute-bound: a large prompt, a tiny generation cap
    prefill = next(c for c in text_cells() if c.name == "prefill")
    assert prefill.max_tokens <= 16
    assert len(prefill.prompt) > 4000          # ~thousands of tokens of context


def test_decode_is_short_prompt_long_generation():
    # decode must be memory-bound: a small prompt, a large generation cap
    decode = next(c for c in text_cells() if c.name == "decode")
    assert decode.max_tokens >= 512
    assert len(decode.prompt) < 500


def test_balanced_is_between_the_two():
    balanced = next(c for c in text_cells() if c.name == "balanced")
    assert 64 <= balanced.max_tokens <= 512


def test_cells_are_deterministic():
    # standardized battery: identical across runs (version-stamped into C2's profile)
    assert [(c.name, c.prompt, c.max_tokens) for c in text_cells()] == \
           [(c.name, c.prompt, c.max_tokens) for c in text_cells()]


def _mock_chat(body, status=200):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = __import__("json").loads(request.content.decode())
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


_CHAT_BODY = {"choices": [{"message": {"content": "acknowledged"}}],
              "usage": {"prompt_tokens": 4123, "completion_tokens": 8}}


def test_http_load_client_posts_chat_and_reads_usage():
    client, seen = _mock_chat(_CHAT_BODY)
    r = HttpLoadClient("http://up", client=client).chat("m1", "hello", max_tokens=8)
    assert r == ChatResult(tok_in=4123, tok_out=8)
    assert seen["url"] == "http://up/v1/chat/completions"
    assert seen["json"]["model"] == "m1"
    assert seen["json"]["max_tokens"] == 8
    assert seen["json"]["stream"] is False
    assert seen["json"]["messages"] == [{"role": "user", "content": "hello"}]


def test_http_load_client_raises_on_http_error():
    client, _ = _mock_chat({"error": "boom"}, status=500)
    with pytest.raises(httpx.HTTPStatusError):
        HttpLoadClient("http://up", client=client).chat("m1", "hi", max_tokens=8)


def test_http_load_client_satisfies_protocol():
    client, _ = _mock_chat(_CHAT_BODY)
    assert isinstance(HttpLoadClient("http://up", client=client), LoadClient)


def test_fake_load_client_returns_fixed_tokens_and_advances_clock():
    class _Clock:
        def __init__(self): self.t = 0.0
        def monotonic(self): return self.t
        def sleep(self, s): self.t += s

    clk = _Clock()
    fake = FakeLoadClient(tok_in=10, tok_out=20, latency_s=2.0, clock=clk)
    assert fake.chat("m", "p", 8) == ChatResult(tok_in=10, tok_out=20)
    assert clk.t == 2.0          # a request consumed 2s of (fake) wall time
    assert isinstance(fake, LoadClient)
