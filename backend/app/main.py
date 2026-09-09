from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api import auth, evaluation, health, jobs, keywords, operators, results, settings
from app.auth.bootstrap import ensure_admin
from app.core.config import get_settings
from app.core.logging import configure_logging, redact_text
from app.core.middleware import CSRFMiddleware, RequestSizeLimitMiddleware, SecurityHeadersMiddleware
from app.database.session import AsyncSessionFactory


settings_object = get_settings()
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings_object.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    (settings_object.STORAGE_ROOT / "recordings").mkdir(parents=True, exist_ok=True)
    (settings_object.STORAGE_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    async with AsyncSessionFactory() as session:
        await ensure_admin(session, settings_object)
    yield


app = FastAPI(
    title="Yeastar Call Analyzer",
    version="1.0.0",
    docs_url=None if settings_object.APP_ENV == "production" else "/docs",
    redoc_url=None,
    openapi_url=None if settings_object.APP_ENV == "production" else "/openapi.json",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings_object.origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token", "Range", "Idempotency-Key"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Disposition"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings_object.trusted_hosts)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings_object.MAX_REQUEST_BYTES)
app.add_middleware(CSRFMiddleware)
app.add_middleware(SecurityHeadersMiddleware, settings=settings_object)

for api_router in (
    auth.router,
    health.router,
    operators.router,
    keywords.router,
    settings.router,
    jobs.router,
    results.router,
    evaluation.features_router,
    evaluation.router,
):
    app.include_router(api_router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    _request: Request,
    exc: RequestValidationError,
) -> ORJSONResponse:
    # FastAPI's default payload includes the rejected input. Omit it so a
    # malformed credential field can never be reflected to the browser.
    errors = [
        {
            "type": str(item.get("type", "value_error")),
            "loc": [str(part) for part in item.get("loc", ())],
            "msg": redact_text(str(item.get("msg", "Invalid request value."))),
        }
        for item in exc.errors()
    ]
    return ORJSONResponse(
        {"detail": errors},
        status_code=422,
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> ORJSONResponse:
    logger.exception("Unhandled request failure: %s", type(exc).__name__)
    return ORJSONResponse({"detail": "An unexpected error occurred."}, status_code=500)
