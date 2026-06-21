from tokenwatt.usage import usage_from_response_json, SelfCounter


def test_usage_from_backend_response():
    body = {"usage": {"prompt_tokens": 12, "completion_tokens": 34,
                      "prompt_tokens_details": {"cached_tokens": 4}}}
    u = usage_from_response_json(body)
    assert (u.input, u.output, u.cached) == (12, 34, 4)
    assert u.source == "backend" and u.confidence == "high"


def test_usage_from_response_none_when_absent():
    assert usage_from_response_json({"choices": []}) is None


def test_self_counter_counts_streamed_content():
    req = {"messages": [{"role": "user", "content": "hello world"}]}
    sc = SelfCounter(req)
    # two SSE data chunks carrying content deltas, then [DONE]
    sc.feed(b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n')
    sc.feed(b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n')
    sc.feed(b'data: [DONE]\n\n')
    u = sc.result()
    assert u.output >= 1 and u.input >= 1
    assert u.source == "self-count" and u.confidence == "low"


def test_self_counter_prefers_backend_usage_chunk():
    # A streamed response carrying an OpenAI usage chunk (stream_options.include_usage):
    # the backend's authoritative counts must win over self-counting.
    req = {"messages": [{"role": "user", "content": "hi"}]}
    sc = SelfCounter(req)
    sc.feed(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n')
    sc.feed(b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}\n\n')
    sc.feed(b'data: [DONE]\n\n')
    u = sc.result()
    assert (u.input, u.output) == (11, 7)
    assert u.source == "backend" and u.confidence == "high"


def test_self_counter_counts_reasoning_content_when_no_usage_chunk():
    # Reasoning models stream their thinking as reasoning_content deltas; with no usage
    # chunk, self-count must include them (else a <think> stream counts as 0 output).
    req = {"messages": [{"role": "user", "content": "hi"}]}
    sc = SelfCounter(req)
    sc.feed(b'data: {"choices":[{"delta":{"reasoning_content":"let me think hard about this"}}]}\n\n')
    sc.feed(b'data: {"choices":[{"delta":{"content":"42"}}]}\n\n')
    sc.feed(b'data: [DONE]\n\n')
    u = sc.result()
    assert u.source == "self-count" and u.confidence == "low"
    assert u.output >= 2   # counts BOTH reasoning_content and content


def test_self_count_includes_tool_calls_when_no_usage_chunk():
    sc = SelfCounter({"messages": [{"role": "user", "content": "hi"}]})
    sc.feed(b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}]}}]}\n')
    sc.feed(b'data: [DONE]\n')
    r = sc.result()
    assert r.source == "self-count"     # no usage chunk arrived -> fallback path
    assert r.output > 0                 # the tool-call name+arguments were counted, not dropped


def test_self_count_reassembles_partial_line_across_chunks():
    # an SSE data line split mid-JSON across two feed() calls must still be counted
    # (SelfCounter._buf holds the partial line until its newline arrives)
    sc = SelfCounter({"messages": [{"role": "user", "content": "hi"}]})
    sc.feed(b'data: {"choices":[{"delta":{"content":"hel')
    sc.feed(b'lo world"}}]}\n')
    sc.feed(b'data: [DONE]\n')
    r = sc.result()
    assert r.source == "self-count"
    assert r.output > 0                 # "hello world" counted despite the chunk boundary
