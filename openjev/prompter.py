"""Сборка компактного пользовательского промпта и обёртка Instruct / ChatML."""

from __future__ import annotations

from enum import Enum
from typing import Any

from openjev.schemas import Question, QuestionChoice, QuestionNoul, QuestionScore

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_SYSTEM = (
    "You are a deterministic decision engine. "
    "Reply with a single option letter. Do not explain."
)

NOUL_TRUE_DEFAULT = "True / Yes — the condition holds given the state"
NOUL_FALSE_DEFAULT = "False / No — the condition does not hold given the state"


class PromptFormat(str, Enum):
    """Целевой шаблон разметки диалога."""

    AUTO = "auto"
    CHATML = "chatml"
    INSTRUCT = "instruct"


def index_to_letter(index: int) -> str:
    """Вернуть букву-маркер варианта (0 → A)."""
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"option index {index} exceeds supported letter range A–Z")
    return LETTERS[index]


def _noul_pole(criteria: dict[str, str] | None, *names: str, default: str) -> str:
    """Найти описание полюса true/false без учёта регистра ключа."""
    if not criteria:
        return default
    lowered = {str(key).strip().lower(): value for key, value in criteria.items()}
    for name in names:
        if name in lowered and str(lowered[name]).strip():
            return str(lowered[name]).strip()
    return default


def option_rows(question: Question) -> list[tuple[str, str]]:
    """Пары (внутренний ключ ответа, текст опции) в порядке отображения A, B, C, …

    * ``choice`` — ключ из ``criteria``;
    * ``score`` — строковый индекс ``\"0\"``, ``\"1\"``, …;
    * ``noul`` — ``\"true\"``, ``\"false\"``.
    """
    if isinstance(question, QuestionChoice):
        rows: list[tuple[str, str]] = []
        for key, description in question.criteria.items():
            desc = str(description).strip()
            text = f"{key} — {desc}" if desc else key
            rows.append((key, text))
        return rows

    if isinstance(question, QuestionScore):
        return [(str(i), f"{i} — {desc.strip()}") for i, desc in enumerate(question.criteria)]

    if isinstance(question, QuestionNoul):
        true_text = _noul_pole(question.criteria, "true", "yes", default=NOUL_TRUE_DEFAULT)
        false_text = _noul_pole(question.criteria, "false", "no", default=NOUL_FALSE_DEFAULT)
        return [("true", true_text), ("false", false_text)]

    raise TypeError(f"unsupported question type: {type(question)!r}")


def build_user_payload(state: str, question: Question) -> str:
    """Собрать компактное тело user-сообщения без chat-обёртки.

    Формат::

        State:
        {state}

        Task: {instructions}
        Options:
        A: ...
        B: ...
        Select single option letter:
    """
    rows = option_rows(question)
    option_lines = [f"{index_to_letter(i)}: {text}" for i, (_, text) in enumerate(rows)]
    options = "\n".join(option_lines)
    return (
        f"State:\n{state.strip()}\n\n"
        f"Task: {question.instructions.strip()}\n"
        f"Options:\n{options}\n"
        "Select single option letter:"
    )


def wrap_chatml(user_payload: str, system: str = DEFAULT_SYSTEM) -> str:
    """Ручной ChatML (Qwen / ChatML-совместимые модели) с префиксом assistant."""
    sys = system.strip()
    user = user_payload.strip()
    return (
        f"<|im_start|>system\n{sys}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def wrap_instruct(user_payload: str, system: str = DEFAULT_SYSTEM) -> str:
    """Универсальный Instruct-шаблон (Alpaca-подобный) без chat_template."""
    return (
        f"### System:\n{system.strip()}\n\n"
        f"### Instruction:\n{user_payload.strip()}\n\n"
        "### Response:\n"
    )


def resolve_format(prompt_format: PromptFormat | str, tokenizer: Any | None) -> PromptFormat:
    """AUTO → CHATML, если у токенизатора есть ``chat_template``, иначе INSTRUCT."""
    fmt = PromptFormat(prompt_format)
    if fmt is not PromptFormat.AUTO:
        return fmt
    template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if template:
        return PromptFormat.CHATML
    return PromptFormat.INSTRUCT


def render_prompt(
    state: str,
    question: Question,
    *,
    tokenizer: Any | None = None,
    prompt_format: PromptFormat | str = PromptFormat.AUTO,
    system: str = DEFAULT_SYSTEM,
) -> str:
    """Полный промпт под last-token logits: chat-обёртка + generation prompt.

    Если токенизатор предоставляет ``apply_chat_template``, он используется
    (корректнее спецтокены конкретной модели). Иначе — ручной ChatML / Instruct.
    """
    user_payload = build_user_payload(state, question)
    fmt = resolve_format(prompt_format, tokenizer)

    apply = getattr(tokenizer, "apply_chat_template", None) if tokenizer is not None else None
    if callable(apply) and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_payload},
        ]
        try:
            return apply(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Не все шаблоны принимают system role — деградируем на ручной формат.
            pass

    if fmt is PromptFormat.CHATML:
        return wrap_chatml(user_payload, system=system)
    return wrap_instruct(user_payload, system=system)


def render_batch(
    state: str,
    questions: dict[str, Question],
    *,
    tokenizer: Any | None = None,
    prompt_format: PromptFormat | str = PromptFormat.AUTO,
    system: str = DEFAULT_SYSTEM,
) -> dict[str, str]:
    """Промпт на каждый вопрос при общем ``state`` (порядок ключей сохраняется)."""
    return {
        key: render_prompt(
            state,
            question,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            system=system,
        )
        for key, question in questions.items()
    }
