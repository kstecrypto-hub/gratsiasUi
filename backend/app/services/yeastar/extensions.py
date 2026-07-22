from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.services.yeastar.errors import YeastarAPIError


QueryValue = str | int | float | bool


class YeastarGetClient(Protocol):
    async def get(
        self,
        endpoint: str,
        *,
        version: str = "v1",
        params: Mapping[str, QueryValue] | None = None,
    ) -> Mapping[str, object]: ...


class ExtensionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | int
    number: str | int
    caller_id_name: str | None = None
    email_addr: str | None = None
    mobile_number: str | None = None
    presence_status: str | None = None

    def provider_dict(self) -> dict[str, object]:
        result = self.model_dump()
        result["id"] = str(self.id)
        result["number"] = str(self.number)
        return result


class ExtensionPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[ExtensionRecord] = Field(default_factory=list)
    total_number: int = Field(default=0, ge=0)


class YeastarExtensions:
    def __init__(self, client: YeastarGetClient, page_size: int = 500) -> None:
        if not 1 <= page_size <= 10_000:
            raise ValueError("page_size must be between 1 and 10000")
        self.client = client
        self.page_size = page_size

    async def page(self, page: int, *, page_size: int | None = None) -> ExtensionPage:
        requested_size = page_size or self.page_size
        if page < 1 or not 1 <= requested_size <= 10_000:
            raise ValueError("Invalid extension pagination")
        payload = await self.client.get(
            "extension/list",
            version="v1",
            params={
                "page": page,
                "page_size": requested_size,
                "sort_by": "number",
                "order_by": "asc",
            },
        )
        try:
            result = ExtensionPage.model_validate(payload)
        except ValueError as exc:
            raise YeastarAPIError("Phone system returned an invalid extension list.") from exc
        if result.total_number == 0 and result.data:
            result.total_number = len(result.data)
        return result

    async def verify_read_access(self) -> None:
        await self.page(1, page_size=1)

    async def list_all(self) -> list[ExtensionRecord]:
        records: list[ExtensionRecord] = []
        page_number = 1
        while True:
            page = await self.page(page_number)
            records.extend(page.data)
            total = page.total_number or len(records)
            if len(records) >= total:
                return records
            if not page.data or page_number >= 10_000:
                raise YeastarAPIError("Phone system returned invalid extension pagination.")
            page_number += 1
