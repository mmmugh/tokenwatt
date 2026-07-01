from tokenwatt.battery import LoadCell, text_cells


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
