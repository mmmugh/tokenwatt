from tokenwatt.reqtype import classify_request


def test_embeddings_path_is_embedding():
    assert classify_request("embeddings", {"input": "hi"}) == "embedding"
    assert classify_request("v1/embeddings", {"input": "hi"}) == "embedding"


def test_image_content_part_is_vision():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    assert classify_request("chat/completions", body) == "vision"


def test_plain_chat_is_text():
    assert classify_request("chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]}) == "text"


def test_list_content_without_image_is_text():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert classify_request("chat/completions", body) == "text"
