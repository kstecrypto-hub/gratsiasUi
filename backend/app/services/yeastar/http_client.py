from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol, TypeVar, cast
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.core.logging import register_secret, sanitized_endpoint
from app.services.yeastar.error_mapping import map_http_status, map_yeastar_error
from app.services.yeastar.errors import (
    YeastarConfigurationError,
    YeastarError,
    YeastarNetworkError,
    YeastarResponseError,
    YeastarTemporarilyUnavailableError,
    YeastarTokenExpiredError,
    YeastarTokenRefreshError,
)
from app.services.yeastar.schemas import AuthTrigger, ConnectionState


logger = logging.getLogger(__name__)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
ModelT = TypeVar("ModelT", bound=BaseModel)


class TokenManagerProtocol(Protocol):
    async def get_access_token(self, trigger: AuthTrigger | None = None) -> str: ...

    async def refresh_access_token(self) -> str: ...

    async def invalidate_access_token(
        self, expected_access_token: str | None = None
    ) -> bool: ...

    async def open_circuit(
        self, reason: ConnectionState, *, last_errcode: int | None = None
    ) -> None: ...


_TRANSIENT_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
)
_TRANSIENT_HTTP_STATUSES = frozenset({502, 503, 504})


def _safe_relative_path(path: str) -> str:
    decoded = path
    for _ in range(3):
        updated = unquote(decoded)
        if updated == decoded:
            break
        decoded = updated
    parsed = urlsplit(decoded)
    if (
        not decoded.startswith("/")
        or decoded.startswith("//")
        or parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or "\\" in decoded
        or ".." in decoded.split("/")
    ):
        raise YeastarConfigurationError("Phone-system API path is invalid.")
    return decoded


class YeastarHttpClient:
    """One pooled Yeastar transport and the sole bounded retry owner."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = http_client
        self._owns_client = http_client is None
        register_secret(self.settings.YEASTAR_CLIENT_SECRET)

    def _require_local_configuration(self) -> None:
        errors = self.settings.yeastar_configuration_errors
        if errors:
            raise YeastarConfigurationError(errors[0]["message"])

    async def _ensure_client(self) -> httpx.AsyncClient:
        self._require_local_configuration()
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=str(self.settings.YEASTAR_BASE_URL).rstrip("/"),
                headers={
                    "User-Agent": self.settings.YEASTAR_USER_AGENT.strip(),
                    "Accept": "application/json",
                },
                verify=self.settings.yeastar_verify_ssl,
                follow_redirects=False,
                timeout=httpx.Timeout(
                    connect=self.settings.yeastar_connect_timeout_seconds,
                    read=self.settings.yeastar_read_timeout_seconds,
                    write=self.settings.yeastar_read_timeout_seconds,
                    pool=self.settings.yeastar_connect_timeout_seconds,
                ),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def __aenter__(self) -> "YeastarHttpClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, JsonValue] | None = None,
        json_body: JsonObject | None = None,
        transient_retry: bool = True,
        require_success: bool = True,
        timeout: httpx.Timeout | float | None = None,
    ) -> JsonObject:
        safe_path = _safe_relative_path(path)
        client = await self._ensure_client()
        retry_limit = self.settings.yeastar_transient_retry_count if transient_retry else 0
        correlation_id = uuid4().hex
        for attempt in range(retry_limit + 1):
            started = time.monotonic()
            response: httpx.Response | None = None
            try:
                response = await client.request(
                    method.upper(),
                    safe_path,
                    params=params,
                    json=json_body,
                    headers={
                        "User-Agent": self.settings.YEASTAR_USER_AGENT.strip(),
                        "Accept": "application/json",
                    },
                    timeout=timeout,
                )
                duration_ms = round((time.monotonic() - started) * 1000)
                if response.status_code in _TRANSIENT_HTTP_STATUSES:
                    if attempt < retry_limit:
                        await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    raise YeastarTemporarilyUnavailableError(
                        "The phone system is temporarily unavailable.",
                        response.status_code,
                    )
                if response.status_code >= 300:
                    raise map_http_status(response.status_code)
                try:
                    raw_payload = response.json()
                except ValueError as exc:
                    raise YeastarResponseError(
                        "The phone system returned an invalid response."
                    ) from exc
                if not isinstance(raw_payload, dict):
                    raise YeastarResponseError(
                        "The phone system returned an invalid response."
                    )
                payload = cast(JsonObject, raw_payload)
                if require_success:
                    if "errcode" not in payload:
                        raise YeastarResponseError(
                            "The phone system returned an invalid response."
                        )
                    code = payload.get("errcode")
                    try:
                        normalized_code = int(cast(int | str, code))
                    except (TypeError, ValueError) as exc:
                        raise YeastarResponseError(
                            "The phone system returned an invalid response."
                        ) from exc
                    if normalized_code != 0:
                        raise map_yeastar_error(
                            normalized_code,
                            cast(str | None, payload.get("errmsg")),
                        )
                logger.info(
                    "%s %s completed http_status=%s errcode=%s duration_ms=%s correlation_id=%s",
                    method.upper(),
                    sanitized_endpoint(safe_path),
                    response.status_code,
                    payload.get("errcode", "none"),
                    duration_ms,
                    correlation_id,
                )
                return payload
            except _TRANSIENT_TRANSPORT_ERRORS as exc:
                if attempt < retry_limit:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                raise YeastarNetworkError(
                    "Could not reach the phone system."
                ) from exc
            except YeastarError as exc:
                logger.warning(
                    "%s %s failed category=%s http_status=%s duration_ms=%s correlation_id=%s",
                    method.upper(),
                    sanitized_endpoint(safe_path),
                    exc.category,
                    response.status_code if response is not None else "none",
                    round((time.monotonic() - started) * 1000),
                    correlation_id,
                )
                raise
        raise AssertionError("unreachable")

    async def request_model(
        self,
        model: type[ModelT],
        method: str,
        path: str,
        **kwargs: object,
    ) -> ModelT:
        try:
            payload = await self.request_json(method, path, **kwargs)  # type: ignore[arg-type]
            return model.model_validate(payload)
        except ValidationError as exc:
            raise YeastarResponseError(
                "The phone system returned an invalid response."
            ) from exc

    async def get(
        self,
        path: str,
        *,
        params: dict[str, JsonValue] | None = None,
        transient_retry: bool = True,
    ) -> JsonObject:
        return await self.request_json(
            "GET", path, params=params, transient_retry=transient_retry
        )

    async def get_with_token(
        self,
        path: str,
        access_token: str,
        *,
        params: dict[str, JsonValue] | None = None,
        transient_retry: bool = True,
    ) -> JsonObject:
        register_secret(access_token)
        request_params = dict(params or {})
        request_params["access_token"] = access_token
        return await self.request_json(
            "GET", path, params=request_params, transient_retry=transient_retry
        )

    async def protected_request(
        self,
        method: str,
        path: str,
        *,
        token_manager: TokenManagerProtocol,
        trigger: AuthTrigger | None = None,
        params: dict[str, JsonValue] | None = None,
        json_body: JsonObject | None = None,
    ) -> JsonObject:
        token = await token_manager.get_access_token(trigger)
        register_secret(token)
        request_params = dict(params or {})
        request_params["access_token"] = token
        try:
            return await self.request_json(
                method, path, params=request_params, json_body=json_body
            )
        except YeastarTokenExpiredError:
            await token_manager.invalidate_access_token(token)
        except YeastarError as exc:
            if exc.opens_circuit:
                await token_manager.open_circuit(
                    exc.connection_state, last_errcode=exc.errcode
                )
            raise
        refreshed_token = await token_manager.refresh_access_token()
        register_secret(refreshed_token)
        request_params["access_token"] = refreshed_token
        try:
            return await self.request_json(
                method, path, params=request_params, json_body=json_body
            )
        except YeastarTokenExpiredError as exc:
            await token_manager.open_circuit(
                ConnectionState.TOKEN_REFRESH_FAILED, last_errcode=10004
            )
            raise YeastarTokenRefreshError(
                "The phone-system session could not be renewed.", 10004
            ) from exc
        except YeastarError as exc:
            if exc.opens_circuit:
                await token_manager.open_circuit(
                    exc.connection_state, last_errcode=exc.errcode
                )
            raise

    @asynccontextmanager
    async def stream(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, JsonValue] | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        safe_path = _safe_relative_path(path)
        client = await self._ensure_client()
        retry_limit = self.settings.yeastar_transient_retry_count
        request_headers = dict(headers or {})
        request_headers["User-Agent"] = self.settings.YEASTAR_USER_AGENT.strip()
        for attempt in range(retry_limit + 1):
            try:
                async with client.stream(
                    method.upper(),
                    safe_path,
                    params=params,
                    headers=request_headers,
                    timeout=timeout
                    or httpx.Timeout(
                        connect=self.settings.yeastar_connect_timeout_seconds,
                        read=self.settings.yeastar_download_timeout_seconds,
                        write=self.settings.yeastar_read_timeout_seconds,
                        pool=self.settings.yeastar_connect_timeout_seconds,
                    ),
                    follow_redirects=False,
                ) as response:
                    if response.status_code in _TRANSIENT_HTTP_STATUSES and attempt < retry_limit:
                        await response.aread()
                        await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    if 300 <= response.status_code:
                        raise map_http_status(response.status_code)
                    yield response
                    return
            except _TRANSIENT_TRANSPORT_ERRORS as exc:
                if attempt < retry_limit:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                raise YeastarNetworkError("Could not reach the phone system.") from exc


__all__ = [
    "JsonObject",
    "JsonValue",
    "TokenManagerProtocol",
    "YeastarHttpClient",
]
