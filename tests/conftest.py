import asyncio
import json
import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route


@pytest.fixture
def fake_upstream_streaming():
    """An OpenAI-ish upstream that streams two content chunks then [DONE], no usage."""
    async def chat(request):
        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.fixture
def fake_upstream_json():
    """A non-streaming upstream that returns a usage block."""
    async def chat(request):
        return JSONResponse({
            "choices": [{"message": {"content": "Hello world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })
    return Starlette(routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", chat, methods=["POST"]),
    ])


@pytest.fixture
def fake_upstream_streaming_with_usage():
    """Streams content chunks AND a final usage chunk (stream_options.include_usage)."""
    async def chat(request):
        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":9}}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.fixture
def fake_upstream_slow_first_chunk():
    """Streams a content chunk after a delay (simulating a cold model load), then [DONE]."""
    async def chat(request):
        async def gen():
            await asyncio.sleep(0.4)   # "load" delay before the first token
            yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield b'data: [DONE]\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream")
    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])
