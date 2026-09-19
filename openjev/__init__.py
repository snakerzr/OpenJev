"""OpenJev: System One Decisions API (0-token typed inference)."""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"

__all__ = [
    "ChoiceResponse",
    "DecisionEngine",
    "DecisionRequest",
    "DecisionResponse",
    "NoulResponse",
    "Question",
    "QuestionChoice",
    "QuestionNoul",
    "QuestionScore",
    "ScoreResponse",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Ленивый экспорт тяжёлых подмодулей."""
    if name == "DecisionEngine":
        from openjev.engine import DecisionEngine

        return DecisionEngine
    if name in {
        "ChoiceResponse",
        "DecisionRequest",
        "DecisionResponse",
        "NoulResponse",
        "Question",
        "QuestionChoice",
        "QuestionNoul",
        "QuestionScore",
        "ScoreResponse",
    }:
        from openjev import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
