"""Evaluation routes in the existing authenticated API; all writes are local references."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.core.config import Settings, get_settings
from app.database.session import get_db
from app.models import Keyword, KeywordCategory
from app.schemas.evaluation import ReferenceDraft, SaveReference, VerifyReference
from app.services.evaluation import AUDIO_TYPES, EvaluationStore, unavailable


def require_evaluation(settings: Annotated[Settings, Depends(get_settings)]) -> EvaluationStore:
    if not settings.EVALUATION_UI_ENABLED:
        raise unavailable("Evaluation is unavailable.")
    return EvaluationStore(settings)


Store = Annotated[EvaluationStore, Depends(require_evaluation)]
features_router = APIRouter(tags=["features"])
router = APIRouter(prefix="/evaluation", tags=["evaluation"],
                   dependencies=[Depends(require_evaluation)])


@features_router.get("/features")
def features(user: CurrentUser, settings: Annotated[Settings, Depends(get_settings)]):
    return {"evaluation_ui_enabled": settings.EVALUATION_UI_ENABLED}


@router.get("")
def list_evaluation(
    user: CurrentUser, store: Store,
    filter: Literal["all", "dev", "test", "unverified", "verified"] = "all",
):
    return store.listing(filter)


@router.get("/keywords")
async def keyword_catalog(
    user: CurrentUser, db: Annotated[AsyncSession, Depends(get_db)],
):
    # Select only the active catalog; never read KeywordMatch or Transcript.
    rows = (await db.execute(
        select(Keyword.id, Keyword.canonical_phrase, KeywordCategory.name)
        .join(KeywordCategory, Keyword.category_id == KeywordCategory.id)
        .where(Keyword.active.is_(True), Keyword.deleted_at.is_(None),
               KeywordCategory.active.is_(True), KeywordCategory.deleted_at.is_(None))
        .order_by(KeywordCategory.name, Keyword.canonical_phrase)
    )).all()
    return [{"id": str(row.id), "canonical_phrase": row.canonical_phrase,
             "category_name": row.name} for row in rows]


@router.get("/{evaluation_id}")
def evaluation_detail(evaluation_id: str, user: CurrentUser, store: Store):
    return store.detail(evaluation_id)


@router.api_route("/{evaluation_id}/audio", methods=["GET", "HEAD"])
def evaluation_audio(evaluation_id: str, user: CurrentUser, store: Store):
    path = store.audio_path(store.record(evaluation_id))
    return FileResponse(path, media_type=AUDIO_TYPES[path.suffix.lower()],
                        headers={"Cache-Control": "no-store",
                                 "Content-Disposition": 'inline; filename="evaluation-recording"'})


@router.api_route("/{evaluation_id}/audio/channel/{channel}", methods=["GET", "HEAD"])
def evaluation_channel(evaluation_id: str, channel: int, user: CurrentUser, store: Store):
    path = store.channel_preview(evaluation_id, channel)
    return FileResponse(path, media_type="audio/wav",
                        headers={"Cache-Control": "no-store",
                                 "Content-Disposition": 'inline; filename="evaluation-channel.wav"'})


@router.put("/{evaluation_id}/reference")
def save_reference(evaluation_id: str, payload: SaveReference, user: CurrentUser, store: Store):
    return store.write_reference(
        evaluation_id, payload.revision,
        ReferenceDraft.model_validate(payload.model_dump(exclude={"revision"})),
    )


@router.post("/{evaluation_id}/verify")
def verify_reference(evaluation_id: str, payload: VerifyReference, user: CurrentUser, store: Store):
    return store.write_reference(evaluation_id, payload.revision, verify=True)
