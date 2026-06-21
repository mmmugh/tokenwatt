from __future__ import annotations

_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def classify_request(path: str, body: dict) -> str:
    """Classify an OpenAI-style request as 'embedding', 'vision', or 'text'."""
    if path.rstrip("/").endswith("embeddings"):
        return "embedding"
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                    return "vision"
    return "text"
