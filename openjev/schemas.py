"""Pydantic-контракт Decisions API: запрос, вопросы и строго типизированные ответы."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Однотокенные маркеры A–Z; больше вариантов нельзя надёжно снять с last-token logits.
_MAX_OPTIONS = 26


class _StrictModel(BaseModel):
    """Базовая модель с запретом неизвестных полей (ловим опечатки в контракте)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QuestionChoice(_StrictModel):
    """Выбор одного класса из словаря критериев.

    ``criteria``: ключ — метка класса, значение — описание для промпта.
    """

    type: Literal["choice"]
    instructions: str = Field(..., min_length=1, description="Постановка задачи для классификатора.")
    criteria: dict[str, str] = Field(
        ...,
        min_length=2,
        max_length=_MAX_OPTIONS,
        description="Метка класса → текстовое описание. Порядок ключей сохраняется.",
    )

    @field_validator("criteria")
    @classmethod
    def _non_empty_labels(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not str(key).strip() for key in value):
            raise ValueError("choice criteria keys must be non-empty")
        return value


class QuestionScore(_StrictModel):
    """Порядковая шкала: индекс 0 соответствует первому элементу ``criteria``."""

    type: Literal["score"]
    instructions: str = Field(..., min_length=1, description="Постановка задачи скоринга.")
    criteria: list[str] = Field(
        ...,
        min_length=2,
        max_length=_MAX_OPTIONS,
        description="Упорядоченная рубрика от 0 до N-1.",
    )

    @field_validator("criteria")
    @classmethod
    def _non_empty_levels(cls, value: list[str]) -> list[str]:
        if any(not str(item).strip() for item in value):
            raise ValueError("score criteria items must be non-empty strings")
        return value


class QuestionNoul(_StrictModel):
    """Бинарная верификация гипотезы. Ответ — P(true) ∈ [0, 1]."""

    type: Literal["noul"]
    instructions: str = Field(..., min_length=1, description="Гипотеза / условие для проверки.")
    criteria: dict[str, str] | None = Field(
        default=None,
        description=(
            "Опциональные описания полюсов. Ожидаются ключи ``true`` и ``false`` "
            "(допускаются ``yes``/``no``). Иначе используются стандартные формулировки."
        ),
    )

    @model_validator(mode="after")
    def _criteria_size(self) -> QuestionNoul:
        if self.criteria is not None and len(self.criteria) < 1:
            raise ValueError("noul criteria, if provided, must contain at least one entry")
        return self


Question = Annotated[
    Union[QuestionChoice, QuestionScore, QuestionNoul],
    Field(discriminator="type"),
]


class DecisionRequest(_StrictModel):
    """Тело ``POST /v1/decisions``: общее состояние и набор типизированных вопросов."""

    state: str = Field(..., min_length=1, description="Наблюдаемый контекст / документ / тикет.")
    questions: dict[str, Question] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Имя вопроса → спецификация. Все вопросы батчатся в один запрос к vLLM.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "state": "Subject: Double charged $49.99 on my card. Please refund.",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Route this support ticket to the correct team.",
                        "criteria": {
                            "billing": "Payments, invoices, refunds",
                            "bugs": "Crashes and broken functionality",
                            "spam": "Unsolicited ads or phishing",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate irritation from calm to furious.",
                        "criteria": ["Calm", "Annoyed", "Angry"],
                    },
                    "refund": {
                        "type": "noul",
                        "instructions": "Is the user requesting a refund?",
                        "criteria": {
                            "true": "Explicit or implicit refund request",
                            "false": "No refund intent",
                        },
                    },
                },
            }
        },
    )


class ChoiceResponse(_StrictModel):
    """Ответ типа ``choice``: выбранная метка и распределение по всем классам."""

    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="Метка класса с максимальной вероятностью.")
    probabilities: dict[str, float] = Field(
        ...,
        description="Нормализованное распределение P(label) по ключам criteria.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="max(probabilities).")


class ScoreResponse(_StrictModel):
    """Ответ типа ``score``.

    ``score`` — дискретный argmax-индекс рубрики (контракт бенчмарка / TypeSafe API).
    ``expected`` — мат. ожидание ``Σ i · P_i`` на шкале 0..N-1.
    """

    type: Literal["score"] = "score"
    score: int = Field(..., ge=0, description="Argmax-индекс на шкале criteria.")
    expected: float = Field(..., description="Ожидаемый скор Σ i × P_i.")
    probabilities: list[float] = Field(..., min_length=2, description="P(i) для каждого уровня шкалы.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="max(probabilities).")


class NoulResponse(_StrictModel):
    """Ответ типа ``noul``: вероятность, что условие истинно."""

    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0.0, le=1.0, description="P(true) после softmax по {Yes, No} / {A, B}.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="max(P(true), P(false)).",
    )


Answer = Annotated[
    Union[ChoiceResponse, ScoreResponse, NoulResponse],
    Field(discriminator="type"),
]


class DecisionResponse(_StrictModel):
    """Обёртка ответов: ключи совпадают с ``request.questions``."""

    answers: dict[str, Answer]


class HealthResponse(BaseModel):
    """Диагностика готовности инференс-сервера."""

    status: Literal["ok", "starting"]
    backend: Literal["vllm"] = "vllm"
    model: str
    device: str
    dtype: str
    ready: bool
