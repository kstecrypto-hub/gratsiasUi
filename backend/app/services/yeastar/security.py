from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import ParseResult, urlparse
from urllib.parse import urlunparse

from app.services.yeastar.errors import YeastarSecurityError
from app.services.yeastar.recordings import validate_download_resource_path


def effective_port(parsed: ParseResult) -> int:
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def validate_origin(url: str) -> ParseResult:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise YeastarSecurityError("Phone-system address is invalid.")
    if parsed.username or parsed.password or parsed.fragment:
        raise YeastarSecurityError("Phone-system address contains forbidden components.")
    try:
        ip = ipaddress.ip_address(parsed.hostname.strip("[]"))
    except ValueError:
        ip = None
    if ip and (ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified):
        raise YeastarSecurityError("Phone-system address resolves to a forbidden network target.")
    return parsed


async def resolve_host(host: str, port: int) -> frozenset[str]:
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise YeastarSecurityError("Phone-system host could not be resolved.") from exc
    addresses: set[str] = set()
    for record in records:
        value = record[4][0]
        ip = ipaddress.ip_address(value)
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise YeastarSecurityError("Phone-system host resolved to a forbidden target.")
        # Private addresses are intentionally permitted: on-premises PBXs commonly use them.
        addresses.add(str(ip))
    if not addresses:
        raise YeastarSecurityError("Phone-system host did not resolve to an address.")
    return frozenset(addresses)


@dataclass
class PinnedOrigin:
    scheme: str
    hostname: str
    port: int
    addresses: frozenset[str]

    @classmethod
    async def create(cls, base_url: str) -> "PinnedOrigin":
        parsed = validate_origin(base_url)
        port = effective_port(parsed)
        addresses = await resolve_host(parsed.hostname or "", port)
        return cls(parsed.scheme, parsed.hostname or "", port, addresses)

    async def validate_download_url(self, base_url: str, supplied_url: str) -> str:
        """Validate Yeastar's relative temporary recording resource.

        Yeastar download resources are deliberately never accepted as absolute
        URLs, even when they name the configured PBX.  Keeping the provider's
        value relative prevents it from selecting a host, port, query string or
        fragment; the access token is added separately by the HTTP client.
        """
        resource_path = validate_download_resource_path(supplied_url)
        parsed = validate_origin(base_url)
        if (
            parsed.scheme != self.scheme
            or parsed.hostname != self.hostname
            or effective_port(parsed) != self.port
        ):
            raise YeastarSecurityError("Phone-system address changed during the download request.")
        current = await resolve_host(self.hostname, self.port)
        if not current.issubset(self.addresses):
            raise YeastarSecurityError("Phone-system DNS target changed during the download request.")
        return resource_path

    def pinned_url(self, resource_path: str) -> str:
        safe_path = validate_download_resource_path(resource_path)
        address = sorted(self.addresses)[0]
        host = f"[{address}]" if ":" in address else address
        default_port = 443 if self.scheme == "https" else 80
        netloc = host if self.port == default_port else f"{host}:{self.port}"
        return urlunparse((self.scheme, netloc, safe_path, "", "", ""))

    @property
    def host_header(self) -> str:
        default_port = 443 if self.scheme == "https" else 80
        return self.hostname if self.port == default_port else f"{self.hostname}:{self.port}"
