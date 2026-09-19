"""Создание движка inference по типу бэкенда."""

from __future__ import annotations

import os
from typing import Any, Literal

from openjev.errors import EngineError

BackendKind = Literal["torch", "vllm"]


def parse_backend(value: str | None) -> BackendKind:
    name = (value or "torch").strip().lower()
    if name not in ("torch", "vllm"):
        raise EngineError(f"unsupported backend {value!r}; use torch or vllm")
    return name  # type: ignore[return-value]


def backend_from_env() -> BackendKind:
    return parse_backend(os.environ.get("OPENJEV_BACKEND", "torch"))


def create_decision_engine(
    backend: BackendKind,
    *,
    model_id: str,
    temperature: float = 1.0,
    max_length: int = 4096,
    prompt_format: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    vllm_url: str | None = None,
    vllm_timeout: float = 120.0,
) -> Any:
    """Вернуть ``DecisionEngine`` или ``VLLMDecisionEngine``."""
    if backend == "torch":
        from openjev.engine import DecisionEngine

        return DecisionEngine(
            model_id,
            temperature=temperature,
            max_length=max_length,
            prompt_format=prompt_format,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if backend == "vllm":
        from openjev.vllm_engine import VLLMDecisionEngine

        base = vllm_url or os.environ.get("OPENJEV_VLLM_URL", "http://localhost:8001/v1")
        engine = VLLMDecisionEngine(
            model_name=model_id,
            base_url=base,
            temperature=temperature,
            prompt_format=prompt_format,
            timeout=vllm_timeout,
            max_length=max_length,
        )
        try:
            engine.health()
        except EngineError as exc:
            raise EngineError(f"vLLM is not reachable at {base}: {exc}") from exc
        return engine

    raise EngineError(f"unsupported backend {backend!r}")
