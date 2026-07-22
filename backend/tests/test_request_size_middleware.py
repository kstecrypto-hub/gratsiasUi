from __future__ import annotations

import asyncio

import pytest
from starlette.responses import StreamingResponse

from app.core.middleware import RequestSizeLimitMiddleware


@pytest.mark.asyncio
async def test_buffered_request_replay_delegates_stream_disconnect_listener() -> None:
    """A StreamingResponse must block on the real receive after body replay.

    Returning synthetic empty request events forever makes Starlette's disconnect
    listener busy-loop. This test deliberately withholds the response chunk until
    that listener reaches the original receive callable.
    """
    receive_calls = 0
    disconnect_listener_waiting = asyncio.Event()
    never_disconnects = asyncio.Event()
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            return {"type": "http.request", "body": b"request", "more_body": False}
        disconnect_listener_waiting.set()
        await never_disconnects.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    async def streaming_app(scope: dict, replay_receive, response_send) -> None:
        async def body():
            await disconnect_listener_waiting.wait()
            yield b"response"

        await StreamingResponse(body())(scope, replay_receive, response_send)

    middleware = RequestSizeLimitMiddleware(streaming_app, max_bytes=1024)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }

    await asyncio.wait_for(middleware(scope, receive, send), timeout=1)

    assert receive_calls == 2
    assert any(message.get("body") == b"response" for message in sent)
