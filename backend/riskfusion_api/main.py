"""RiskFusion API. Run: ``uvicorn riskfusion_api.main:app``. OpenAPI at /docs and /openapi.json."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import riskfusion

from .routers import participants, sessions, system
from .services.lifecycle import InvalidTransition
from .settings import get_settings

log = logging.getLogger("riskfusion.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RiskFusion API",
        version=riskfusion.__version__,
        description=(
            "Session management, consent, mock recording and (in later phases) calibrated integrity-risk "
            "assessment. The system produces review recommendations only — never verdicts."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(InvalidTransition)
    async def _transition(_: Request, exc: InvalidTransition) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors: list[dict[str, Any]] = [
            {
                "field": ".".join(str(p) for p in e.get("loc", ())[1:]),
                "message": str(e.get("msg", "")).removeprefix("Value error, "),
            }
            for e in exc.errors()
        ]
        message = errors[0]["message"] if errors else "The request was not valid."
        return JSONResponse(status_code=422, content={"detail": message, "errors": errors})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong on the server. Try again.", "technical": type(exc).__name__},
        )

    app.include_router(system.router)
    app.include_router(participants.router)
    app.include_router(sessions.router)
    return app


app = create_app()
