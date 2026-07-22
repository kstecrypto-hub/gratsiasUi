from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.services.yeastar.http_client import YeastarHttpClient


def configured_settings() -> Settings:
    return Settings(
        _env_file=None,
        APP_ENV="test",
        APP_SECRET_KEY="s" * 32,
        YEASTAR_BASE_URL="https://pbx.internal",
        YEASTAR_CLIENT_ID="client-id",
        YEASTAR_CLIENT_SECRET="client-secret",
        YEASTAR_TRANSIENT_RETRY_COUNT=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.WriteError, httpx.WriteTimeout])
async def test_write_side_transport_failure_has_exactly_one_retry(
    error_type: type[httpx.TransportError],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("simulated write-side connection reset", request=request)
        return httpx.Response(200, json={"errcode": 0, "data": {"ok": True}})

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.services.yeastar.http_client.asyncio.sleep", no_sleep)
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://pbx.internal", transport=transport
    ) as raw_client:
        client = YeastarHttpClient(configured_settings(), raw_client)
        result = await client.get("/openapi/v1.0/system/information")

    assert result["errcode"] == 0
    assert calls == 2
