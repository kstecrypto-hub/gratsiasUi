from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.time import utc_now
from app.models import Operator, SyncRun
from app.models.enums import RunStatus, SyncType
from app.services.yeastar.client import YeastarClient


@dataclass(frozen=True)
class OperatorSyncResult:
    created: int
    updated: int
    total: int
    synchronized_at: datetime


async def synchronize_operators(
    session: AsyncSession, client: YeastarClient
) -> OperatorSyncResult:
    lock = client.redis.lock("yca:operator-sync", timeout=180, blocking_timeout=60)
    async with lock:
        completed = asyncio.Event()

        async def operation() -> OperatorSyncResult:
            try:
                return await _synchronize_operators_locked(session, client)
            finally:
                completed.set()

        async def renew_lock() -> None:
            while True:
                try:
                    await asyncio.wait_for(completed.wait(), timeout=60)
                    return
                except TimeoutError:
                    if not await lock.owned():
                        raise RuntimeError("Operator synchronization lock was lost.")
                    await lock.extend(180, replace_ttl=True)

        operation_task = asyncio.create_task(operation())
        renewal_task = asyncio.create_task(renew_lock())
        done, _ = await asyncio.wait(
            {operation_task, renewal_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if renewal_task in done and not operation_task.done():
            renewal_error = renewal_task.exception()
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            if renewal_error is not None:
                raise renewal_error
            raise RuntimeError("Operator synchronization lock renewal stopped.")
        await renewal_task
        return await operation_task


async def _synchronize_operators_locked(
    session: AsyncSession, client: YeastarClient
) -> OperatorSyncResult:
    synchronized_at = utc_now()
    bucket = synchronized_at.strftime("%Y%m%d%H%M")
    idempotency_key = hashlib.sha256(f"operators:{bucket}".encode()).hexdigest()
    existing_run = await session.scalar(
        select(SyncRun).where(SyncRun.idempotency_key == idempotency_key)
    )
    if existing_run and existing_run.status == RunStatus.COMPLETED:
        return OperatorSyncResult(
            existing_run.records_created,
            existing_run.records_updated,
            existing_run.records_seen,
            existing_run.completed_at or synchronized_at,
        )
    run = existing_run or SyncRun(
        sync_type=SyncType.OPERATORS,
        status=RunStatus.RUNNING,
        idempotency_key=idempotency_key,
        started_at=synchronized_at,
    )
    session.add(run)
    run.status = RunStatus.RUNNING
    run.error_category = None
    run.error_message = None
    await session.flush()
    try:
        extensions = await client.list_extensions()
        existing = {
            item.yeastar_extension_id: item
            for item in (await session.scalars(select(Operator))).all()
        }
        created = 0
        updated = 0
        seen = {
            str(extension.get("id") or "").strip()
            for extension in extensions
            if str(extension.get("id") or "").strip()
        }
        # Disable missing IDs first so a newly-created extension can safely reuse its number.
        for provider_id, operator in existing.items():
            if provider_id not in seen and operator.deleted_at is None:
                operator.provider_active = False
                operator.deleted_at = synchronized_at
                updated += 1
        await session.flush()
        for extension in extensions:
            provider_id = str(extension.get("id") or "").strip()
            number = str(extension.get("number") or "").strip()
            if not provider_id or not number:
                run.records_failed += 1
                continue
            display_name = str(extension.get("caller_id_name") or number).strip()[:255]
            email = str(extension.get("email_addr") or "").strip()[:320] or None
            mobile_number = str(
                extension.get("mobile_number") or extension.get("mobile") or ""
            ).strip()[:64] or None
            presence_status = str(
                extension.get("presence_status") or extension.get("presence") or ""
            ).strip()[:64] or None
            operator = existing.get(provider_id)
            if operator is None:
                operator = Operator(
                    yeastar_extension_id=provider_id,
                    extension_number=number,
                    display_name=display_name,
                    email=email,
                    mobile_number=mobile_number,
                    presence_status=presence_status,
                    provider_active=True,
                    enabled=True,
                    last_synced_at=synchronized_at,
                )
                session.add(operator)
                created += 1
            else:
                changed = (
                    operator.extension_number != number
                    or operator.display_name != display_name
                    or operator.email != email
                    or operator.mobile_number != mobile_number
                    or operator.presence_status != presence_status
                    or not operator.provider_active
                    or operator.deleted_at is not None
                )
                operator.extension_number = number
                operator.display_name = display_name
                operator.email = email
                operator.mobile_number = mobile_number
                operator.presence_status = presence_status
                operator.provider_active = True
                operator.last_synced_at = synchronized_at
                operator.deleted_at = None
                updated += int(changed)
        run.status = RunStatus.COMPLETED
        run.records_seen = len(extensions)
        run.records_created = created
        run.records_updated = updated
        run.completed_at = utc_now()
        await session.commit()
        return OperatorSyncResult(created, updated, len(extensions), synchronized_at)
    except Exception as exc:
        await session.rollback()
        run.status = RunStatus.FAILED
        run.error_category = getattr(exc, "category", "unexpected")
        run.error_message = str(exc)[:1000]
        run.completed_at = utc_now()
        session.add(run)
        await session.commit()
        raise
