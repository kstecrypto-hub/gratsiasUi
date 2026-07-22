from __future__ import annotations

import socket

import pytest

from app.services.yeastar import security
from app.services.yeastar.errors import YeastarSecurityError
from app.services.yeastar.security import PinnedOrigin, resolve_host, validate_origin


def _origin() -> PinnedOrigin:
    return PinnedOrigin(
        scheme="https",
        hostname="pbx.example.test",
        port=443,
        addresses=frozenset({"203.0.113.10"}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supplied_path",
    [
        "/api/download/opaque-recording-resource",
        "/api/temporary-resource/recording",
    ],
)
async def test_recording_download_accepts_only_relative_temporary_api_resource(
    monkeypatch: pytest.MonkeyPatch,
    supplied_path: str,
) -> None:
    async def stable_dns(_host: str, _port: int) -> frozenset[str]:
        return frozenset({"203.0.113.10"})

    monkeypatch.setattr(security, "resolve_host", stable_dns)
    origin = _origin()

    resource_path = await origin.validate_download_url(
        "https://pbx.example.test", supplied_path
    )

    assert resource_path == supplied_path
    assert origin.pinned_url(resource_path) == f"https://203.0.113.10{supplied_path}"
    assert origin.host_header == "pbx.example.test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supplied_url",
    [
        "https://pbx.example.test/api/download/recording.wav",
        "https://attacker.example/api/download/recording.wav",
        "//attacker.example/api/download/recording.wav",
        "/api/download/recording.wav?signature=opaque",
        "/api/download/recording.wav#fragment",
        "/api/download/%252e%252e%252fprivate.wav",
        "/api/download/%2e%2e/private.wav",
        "/openapi/v1.0/recording.wav",
    ],
)
async def test_recording_download_rejects_untrusted_resource_values(
    supplied_url: str,
) -> None:
    with pytest.raises(YeastarSecurityError):
        await _origin().validate_download_url("https://pbx.example.test", supplied_url)


@pytest.mark.asyncio
async def test_recording_url_rejects_dns_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rebound_dns(_host: str, _port: int) -> frozenset[str]:
        return frozenset({"203.0.113.11"})

    monkeypatch.setattr(security, "resolve_host", rebound_dns)

    with pytest.raises(YeastarSecurityError, match="DNS target changed"):
        await _origin().validate_download_url(
            "https://pbx.example.test", "/api/download/recording.wav"
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/recording.wav",
        "http://[::1]/recording.wav",
        "https://" + "user:password@" + "pbx.example.test/recording.wav",
        "file:///tmp/recording.wav",
    ],
)
def test_origin_validation_rejects_forbidden_ssrf_targets(url: str) -> None:
    with pytest.raises(YeastarSecurityError):
        validate_origin(url)


@pytest.mark.asyncio
async def test_dns_resolution_rejects_a_hostname_that_resolves_to_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Loop:
        async def getaddrinfo(self, *_: object, **__: object) -> list[tuple[object, ...]]:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", 443),
                )
            ]

    monkeypatch.setattr(security.asyncio, "get_running_loop", lambda: _Loop())

    with pytest.raises(YeastarSecurityError, match="forbidden target"):
        await resolve_host("pbx.example.test", 443)


@pytest.mark.asyncio
async def test_private_on_premises_pbx_address_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Loop:
        async def getaddrinfo(self, *_: object, **__: object) -> list[tuple[object, ...]]:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("192.168.10.20", 443),
                )
            ]

    monkeypatch.setattr(security.asyncio, "get_running_loop", lambda: _Loop())

    assert await resolve_host("pbx.internal", 443) == frozenset({"192.168.10.20"})
