from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.database.base import Base
from app.models import Call, CallLeg, CallParticipant, Operator
from app.workers.pipeline import _upsert_details


@pytest.mark.asyncio
@pytest.mark.parametrize("first,second,expected", [
    (["B", "C"], ["A", "B", "C"], ["A", "B", "C"]),
    (["A", "B"], ["B", "A"], ["B", "A"]),
    (["A", "B"], ["C", "B"], ["C", "B", "A"]),
    (["A"], ["A", "A", "B"], ["A", "B"]),
])
async def test_rediscovery_reorders_legs_without_losing_identities(first, second, expected):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    settings = Settings(APP_ENV="test")
    now = datetime.now(UTC)

    def detail(ids):
        return {"timeline": [
            {"transaction_id": "transaction", "cdr_id": leg_id,
             "call_to_ext_id": "operator", "status": "ANSWERED"}
            for leg_id in ids
        ]}

    try:
        async with sessions() as session:
            call = Call(yeastar_uid="rediscovered-call", started_at=now)
            operator = Operator(yeastar_extension_id="operator", extension_number="101",
                                display_name="Operator", last_synced_at=now)
            session.add_all([call, operator])
            await session.flush()
            await _upsert_details(session, call, detail(first), [operator], settings)
            await session.commit()
            original_legs = {leg.yeastar_leg_id: leg.id for leg in
                             (await session.scalars(select(CallLeg))).all()}
            original_participants = {participant.id for participant in
                                     (await session.scalars(select(CallParticipant))).all()}

            # Repeating the same expanded response must remain idempotent.
            for _ in range(2):
                await _upsert_details(session, call, detail(second), [operator], settings)
                await session.commit()
                legs = (await session.scalars(
                    select(CallLeg).order_by(CallLeg.sequence_number)
                )).all()
                assert [leg.yeastar_leg_id for leg in legs] == [
                    f"transaction:{leg_id}" for leg_id in expected
                ]
                assert [leg.sequence_number for leg in legs] == list(range(1, len(expected) + 1))
                assert all(leg.id == original_legs.get(leg.yeastar_leg_id, leg.id) for leg in legs)
                participants = (await session.scalars(select(CallParticipant))).all()
                assert original_participants <= {participant.id for participant in participants}
                assert len(participants) == len(expected)
    finally:
        await engine.dispose()
