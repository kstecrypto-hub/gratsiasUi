from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.dependencies import CurrentUser
from app.core.time import utc_now
from app.database.session import get_db
from app.models import Keyword, KeywordCategory, KeywordVariant
from app.schemas.common import Page
from app.schemas.keywords import (
    KeywordCategoryCreate,
    KeywordCategoryResponse,
    KeywordCategoryUpdate,
    KeywordCreate,
    KeywordResponse,
    KeywordUpdate,
    KeywordVariantResponse,
)
from app.services.audit import audit
from app.services.keyword_matching.normalization import normalize_greek


router = APIRouter(tags=["keywords"])


def keyword_response(keyword: Keyword) -> KeywordResponse:
    return KeywordResponse(
        id=keyword.id,
        category_id=keyword.category_id,
        canonical_phrase=keyword.canonical_phrase,
        variants=[KeywordVariantResponse(id=item.id, phrase=item.phrase) for item in keyword.variants],
        accent_insensitive=keyword.accent_insensitive,
        whole_word=keyword.whole_word,
        exact_phrase=keyword.exact_phrase,
        fuzzy_match=keyword.fuzzy_match,
        fuzzy_threshold=keyword.fuzzy_threshold / 100,
        active=keyword.active,
        severity=keyword.severity,
        notes=keyword.notes,
        created_at=keyword.created_at,
        updated_at=keyword.updated_at,
    )


@router.get("/keyword-categories", response_model=Page[KeywordCategoryResponse])
async def list_categories(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = 1,
    page_size: int = 100,
) -> Page[KeywordCategoryResponse]:
    page, page_size = max(1, page), min(500, max(1, page_size))
    condition = KeywordCategory.deleted_at.is_(None)
    total = await db.scalar(select(func.count()).select_from(KeywordCategory).where(condition)) or 0
    count_query = (
        select(func.count(Keyword.id))
        .where(Keyword.category_id == KeywordCategory.id, Keyword.deleted_at.is_(None))
        .correlate(KeywordCategory)
        .scalar_subquery()
    )
    rows = (
        await db.execute(
            select(KeywordCategory, count_query.label("keyword_count"))
            .where(condition)
            .order_by(KeywordCategory.name)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    items = [
        KeywordCategoryResponse(
            id=row.id,
            name=row.name,
            description=row.description,
            active=row.active,
            created_at=row.created_at,
            updated_at=row.updated_at,
            keyword_count=count,
        )
        for row, count in rows
    ]
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.post(
    "/keyword-categories", response_model=KeywordCategoryResponse, status_code=status.HTTP_201_CREATED
)
async def create_category(
    payload: KeywordCategoryCreate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeywordCategoryResponse:
    category = KeywordCategory(**payload.model_dump())
    db.add(category)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Category name already exists.") from exc
    await audit(
        db,
        action="keyword_category.create",
        request=request,
        user=user,
        resource_type="keyword_category",
        resource_id=str(category.id),
    )
    await db.commit()
    await db.refresh(category)
    return KeywordCategoryResponse.model_validate(category)


@router.patch("/keyword-categories/{category_id}", response_model=KeywordCategoryResponse)
async def update_category(
    category_id: UUID,
    payload: KeywordCategoryUpdate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeywordCategoryResponse:
    category = await db.get(KeywordCategory, category_id)
    if category is None or category.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(category, key, value)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Category name already exists.") from exc
    await audit(
        db,
        action="keyword_category.update",
        request=request,
        user=user,
        resource_type="keyword_category",
        resource_id=str(category.id),
    )
    await db.commit()
    await db.refresh(category)
    return KeywordCategoryResponse.model_validate(category)


@router.delete("/keyword-categories/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_category(
    category_id: UUID,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    category = await db.get(KeywordCategory, category_id)
    if category is None or category.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")
    now = utc_now()
    category.active = False
    category.deleted_at = now
    keywords = (await db.scalars(select(Keyword).where(Keyword.category_id == category.id))).all()
    for keyword in keywords:
        keyword.active = False
        keyword.deleted_at = now
    await audit(
        db,
        action="keyword_category.delete",
        request=request,
        user=user,
        resource_type="keyword_category",
        resource_id=str(category.id),
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/keywords", response_model=Page[KeywordResponse])
async def list_keywords(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    category_id: UUID | None = None,
    active: bool | None = None,
    page: int = 1,
    page_size: int = 100,
) -> Page[KeywordResponse]:
    page, page_size = max(1, page), min(500, max(1, page_size))
    conditions = [Keyword.deleted_at.is_(None)]
    if category_id:
        conditions.append(Keyword.category_id == category_id)
    if active is not None:
        conditions.append(Keyword.active.is_(active))
    total = await db.scalar(select(func.count()).select_from(Keyword).where(*conditions)) or 0
    rows = (
        await db.scalars(
            select(Keyword)
            .options(selectinload(Keyword.variants))
            .where(*conditions)
            .order_by(Keyword.canonical_phrase)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    return Page(
        items=[keyword_response(row) for row in rows], total=total, page=page, page_size=page_size
    )


async def require_category(db: AsyncSession, category_id: UUID) -> KeywordCategory:
    category = await db.get(KeywordCategory, category_id)
    if category is None or category.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Category not found.")
    return category


def replace_variants(keyword: Keyword, phrases: list) -> None:
    keyword.variants.clear()
    seen: set[str] = set()
    for item in phrases:
        normalized = normalize_greek(item.phrase, remove_accents=keyword.accent_insensitive)
        if normalized and normalized not in seen and normalized != keyword.normalized_phrase:
            keyword.variants.append(KeywordVariant(phrase=item.phrase, normalized_phrase=normalized))
            seen.add(normalized)


@router.post("/keywords", response_model=KeywordResponse, status_code=status.HTTP_201_CREATED)
async def create_keyword(
    payload: KeywordCreate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeywordResponse:
    await require_category(db, payload.category_id)
    keyword = Keyword(
        category_id=payload.category_id,
        canonical_phrase=" ".join(payload.canonical_phrase.split()),
        normalized_phrase=normalize_greek(
            payload.canonical_phrase, remove_accents=payload.accent_insensitive
        ),
        accent_insensitive=payload.accent_insensitive,
        whole_word=payload.whole_word,
        exact_phrase=payload.exact_phrase,
        fuzzy_match=payload.fuzzy_match,
        fuzzy_threshold=round(payload.fuzzy_threshold * 100),
        active=payload.active,
        severity=payload.severity,
        notes=payload.notes,
    )
    replace_variants(keyword, payload.variants)
    db.add(keyword)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Phrase already exists.") from exc
    await audit(
        db,
        action="keyword.create",
        request=request,
        user=user,
        resource_type="keyword",
        resource_id=str(keyword.id),
    )
    await db.commit()
    keyword = await db.scalar(
        select(Keyword).options(selectinload(Keyword.variants)).where(Keyword.id == keyword.id)
    )
    assert keyword is not None
    return keyword_response(keyword)


@router.patch("/keywords/{keyword_id}", response_model=KeywordResponse)
async def update_keyword(
    keyword_id: UUID,
    payload: KeywordUpdate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> KeywordResponse:
    keyword = await db.scalar(
        select(Keyword).options(selectinload(Keyword.variants)).where(Keyword.id == keyword_id)
    )
    if keyword is None or keyword.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Phrase not found.")
    values = payload.model_dump(exclude_unset=True, exclude={"variants", "fuzzy_threshold"})
    if payload.category_id is not None:
        await require_category(db, payload.category_id)
    for key, value in values.items():
        setattr(keyword, key, value)
    if payload.fuzzy_threshold is not None:
        keyword.fuzzy_threshold = round(payload.fuzzy_threshold * 100)
    if payload.canonical_phrase is not None or payload.accent_insensitive is not None:
        keyword.canonical_phrase = " ".join(keyword.canonical_phrase.split())
        keyword.normalized_phrase = normalize_greek(
            keyword.canonical_phrase, remove_accents=keyword.accent_insensitive
        )
    if payload.variants is not None:
        replace_variants(keyword, payload.variants)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Phrase already exists.") from exc
    await audit(
        db,
        action="keyword.update",
        request=request,
        user=user,
        resource_type="keyword",
        resource_id=str(keyword.id),
    )
    await db.commit()
    keyword = await db.scalar(
        select(Keyword).options(selectinload(Keyword.variants)).where(Keyword.id == keyword.id)
    )
    assert keyword is not None
    return keyword_response(keyword)


@router.delete("/keywords/{keyword_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_keyword(
    keyword_id: UUID,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    keyword = await db.get(Keyword, keyword_id)
    if keyword is None or keyword.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Phrase not found.")
    keyword.active = False
    keyword.deleted_at = utc_now()
    await audit(
        db,
        action="keyword.delete",
        request=request,
        user=user,
        resource_type="keyword",
        resource_id=str(keyword.id),
    )
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
