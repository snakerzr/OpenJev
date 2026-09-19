"""FastAPI-приложение OpenJev: ``GET /health`` и ``POST /v1/decisions``."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openjev import __version__
from openjev.engine import DecisionEngine
from openjev.errors import EngineError
from openjev.schemas import DecisionRequest, DecisionResponse, HealthResponse

logger = logging.getLogger("openjev")

DEFAULT_MODEL_ID = os.environ.get("OPENJEV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEFAULT_VLLM_URL = os.environ.get("OPENJEV_VLLM_URL", "http://localhost:8001/v1")


def _settings_from_env() -> dict[str, Any]:
    return {
        "model_id": os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL_ID),
        "vllm_url": os.environ.get("OPENJEV_VLLM_URL", DEFAULT_VLLM_URL),
        "temperature": float(os.environ.get("OPENJEV_TEMPERATURE", "1.0")),
        "max_length": int(os.environ.get("OPENJEV_MAX_LENGTH", "4096")),
        "prompt_format": os.environ.get("OPENJEV_PROMPT_FORMAT", "instruct"),
        "trust_remote_code": os.environ.get("OPENJEV_TRUST_REMOTE_CODE", "0") in {"1", "true", "True"},
        "timeout": float(os.environ.get("OPENJEV_VLLM_TIMEOUT", "120")),
    }


def create_app(
    *,
    model_id: str | None = None,
    vllm_url: str | None = None,
    temperature: float | None = None,
    max_length: int | None = None,
    prompt_format: str | None = None,
    trust_remote_code: bool | None = None,
    timeout: float | None = None,
    engine: DecisionEngine | None = None,
) -> FastAPI:
    """Фабрика приложения. ``DecisionEngine`` создаётся в lifespan, если ``engine`` не передан."""

    env = _settings_from_env()
    resolved_model = model_id or env["model_id"]
    resolved_vllm_url = vllm_url or env["vllm_url"]
    temperature = env["temperature"] if temperature is None else temperature
    max_length = env["max_length"] if max_length is None else max_length
    prompt_format = env["prompt_format"] if prompt_format is None else prompt_format
    trust_remote_code = (
        env["trust_remote_code"] if trust_remote_code is None else trust_remote_code
    )
    timeout = env["timeout"] if timeout is None else timeout

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is not None:
            app.state.engine = engine
            logger.info("Using injected engine (%s)", getattr(engine, "model_id", "?"))
        else:
            logger.info(
                "Starting OpenJev gateway model=%s vllm_url=%s",
                resolved_model,
                resolved_vllm_url,
            )
            try:
                inst = DecisionEngine(
                    resolved_model,
                    resolved_vllm_url,
                    temperature=temperature,
                    max_length=max_length,
                    prompt_format=prompt_format,
                    trust_remote_code=trust_remote_code,
                    timeout=timeout,
                )
                inst.health()
                app.state.engine = inst
            except EngineError as exc:
                logger.exception("Failed to initialize decision engine")
                raise RuntimeError(str(exc)) from exc
        try:
            yield
        finally:
            inst = getattr(app.state, "engine", None)
            if inst is not None and hasattr(inst, "close"):
                inst.close()
            app.state.engine = None

    app = FastAPI(
        title="OpenJev Decisions API",
        description=(
            "System One gateway: типизированные ответы ``choice`` / ``score`` / ``noul`` "
            "через vLLM (logprobs, max_tokens=1)."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    origins = [item.strip() for item in os.environ.get("OPENJEV_CORS_ORIGINS", "*").split(",") if item.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Inference-Time-Ms"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = exc.errors()
        unknown_type = any(
            err.get("type") in {"union_tag_invalid", "union_tag_not_found"}
            or "discriminator" in str(err.get("type", ""))
            or ("expected tags" in str(err.get("msg", "")).lower() and "choice" in str(err.get("msg", "")).lower())
            for err in details
        )
        message = (
            "Unknown or invalid question type. Supported types: choice, score, noul."
            if unknown_type
            else "Request validation failed."
        )
        return JSONResponse(
            status_code=422,
            content={"detail": message, "errors": details},
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        inst = getattr(request.app.state, "engine", None)
        if inst is None:
            return HealthResponse(
                status="starting",
                backend="vllm",
                model=resolved_model,
                device="unknown",
                dtype="unknown",
                ready=False,
            )
        ready = True
        if hasattr(inst, "health"):
            try:
                ready = bool(inst.health().get("ready", True))
            except EngineError:
                ready = False
        return HealthResponse(
            status="ok" if ready else "starting",
            backend="vllm",
            model=getattr(inst, "model_id", resolved_model),
            device=getattr(inst, "device_label", "unknown"),
            dtype=getattr(inst, "dtype_label", "gateway"),
            ready=ready,
        )

    @app.post("/v1/decisions", response_model=DecisionResponse, tags=["decisions"])
    def create_decisions(
        payload: DecisionRequest,
        request: Request,
        response: Response,
    ) -> DecisionResponse:
        inst = getattr(request.app.state, "engine", None)
        if inst is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Decision engine is not ready",
            )

        logger.info(
            "POST /v1/decisions questions=%s state_chars=%d",
            {name: q.type for name, q in payload.questions.items()},
            len(payload.state),
        )

        t0 = time.perf_counter()
        try:
            result = inst.predict(payload.state, payload.questions)
        except EngineError as exc:
            logger.exception("Engine error")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled inference error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal inference error",
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.2f}"
        return result

    return app


app = create_app()
