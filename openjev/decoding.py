"""Общая логика softmax и упаковки ответов из логитов / logprobs."""

from __future__ import annotations

import math
from typing import Sequence

from openjev.errors import EngineError
from openjev.prompter import LETTERS
from openjev.schemas import (
    Answer,
    ChoiceResponse,
    NoulResponse,
    Question,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
    ScoreResponse,
)

LOGPROB_FLOOR = -100.0
_PROB_DIGITS = 6


def letter_aliases(letter: str) -> tuple[str, ...]:
    """Строковые варианты токена буквы для сопоставления с vLLM ``top_logprobs``."""
    return (letter, f" {letter}", f"\n{letter}")


def max_logprob_for_letter(top_logprobs: dict[str, float], letter: str) -> float:
    """Максимальный logprob среди алиасов буквы; иначе ``LOGPROB_FLOOR``."""
    best = LOGPROB_FLOOR
    for alias in letter_aliases(letter):
        if alias in top_logprobs:
            best = max(best, float(top_logprobs[alias]))
    return best


def option_logprobs_from_top(top_logprobs: dict[str, float], num_options: int) -> list[float]:
    """Logprob на каждый вариант A.. по числу опций."""
    if num_options > len(LETTERS):
        raise EngineError(f"too many options ({num_options}); max {len(LETTERS)}")
    return [max_logprob_for_letter(top_logprobs, LETTERS[i]) for i in range(num_options)]


def softmax_from_logprobs(
    logprobs: Sequence[float],
    temperature: float,
    *,
    strict_floor_check: bool = True,
) -> list[float]:
    """Numerically stable softmax по срезу logprobs с температурой."""
    if temperature <= 0:
        raise EngineError("temperature must be > 0")
    if strict_floor_check and logprobs and all(float(lp) <= LOGPROB_FLOOR for lp in logprobs):
        raise EngineError(
            "Prediction degraded: none of candidate tokens found in top_logprobs"
        )
    scaled = [float(lp) / temperature for lp in logprobs]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        raise EngineError("softmax normalization failed")
    return [e / total for e in exps]


def _round_prob(value: float) -> float:
    return float(round(float(value), _PROB_DIGITS))


def _round_dist(values: Sequence[float]) -> list[float]:
    return [_round_prob(v) for v in values]


def pack_answer(
    question: Question,
    labels: tuple[str, ...],
    probabilities: Sequence[float],
) -> Answer:
    """Собрать типизированный ответ из нормализованных вероятностей по вариантам."""
    values = [float(p) for p in probabilities]
    if len(values) != len(labels):
        raise EngineError("labels and probabilities length mismatch")
    best = max(range(len(values)), key=lambda i: values[i])

    if isinstance(question, QuestionChoice):
        dist = {label: _round_prob(p) for label, p in zip(labels, values, strict=True)}
        return ChoiceResponse(
            choice=labels[best],
            probabilities=dist,
            confidence=_round_prob(values[best]),
        )

    if isinstance(question, QuestionScore):
        expected = sum(i * p for i, p in enumerate(values))
        return ScoreResponse(
            score=best,
            expected=_round_prob(expected),
            probabilities=_round_dist(values),
            confidence=_round_prob(values[best]),
        )

    if isinstance(question, QuestionNoul):
        p_true = values[0]
        return NoulResponse(
            noul=_round_prob(p_true),
            confidence=_round_prob(max(p_true, 1.0 - p_true)),
        )

    raise EngineError(f"unsupported question type {type(question)!r}")
