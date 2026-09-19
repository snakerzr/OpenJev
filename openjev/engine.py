"""DecisionEngine: один batched forward-pass и softmax только по токенам-кандидатам."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from openjev.errors import EngineError
from openjev.prompter import LETTERS, option_rows, render_batch
from openjev.schemas import (
    Answer,
    ChoiceResponse,
    DecisionResponse,
    NoulResponse,
    Question,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
    ScoreResponse,
)

logger = logging.getLogger(__name__)

_PROB_DIGITS = 6


@dataclass(frozen=True, slots=True)
class _PreparedItem:
    key: str
    question: Question
    prompt: str
    labels: tuple[str, ...]
    alias_ids: tuple[tuple[int, ...], ...]


def detect_torch_device() -> str:
    """Предпочтительное устройство: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str | torch.dtype | None, device: str) -> torch.dtype:
    """bfloat16 на Ampere+, float16 на остальных GPU/MPS, float32 на CPU."""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = (dtype or "auto").lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name != "auto":
        if name not in mapping:
            raise EngineError(f"unsupported dtype {dtype!r}")
        return mapping[name]
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _round_prob(value: float) -> float:
    return float(round(float(value), _PROB_DIGITS))


def _round_dist(values: Sequence[float]) -> list[float]:
    return [_round_prob(v) for v in values]


class DecisionEngine:
    """Инференс System One: 0 output tokens, logit slicing, калиброванный softmax.

    Все вопросы к одному ``state`` токенизируются с left-padding и прогоняются
    **одним** ``model(...)``. Из ``logits[:, -1, :]`` берётся срез только по
    ID токенов-кандидатов (буквы A, B, C, …), затем ``softmax`` с температурой.
    """

    def __init__(
        self,
        model_id: str,
        *,
        temperature: float = 1.0,
        max_length: int = 4096,
        prompt_format: str = "auto",
        dtype: str | torch.dtype | None = "auto",
        device_map: str | dict[str, Any] | None = "auto",
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        if temperature <= 0:
            raise EngineError("temperature must be > 0")
        if max_length < 32:
            raise EngineError("max_length must be >= 32")

        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_length = int(max_length)
        self.prompt_format = prompt_format
        self.trust_remote_code = trust_remote_code

        self.device_kind = detect_torch_device()
        self.torch_dtype = resolve_dtype(dtype, self.device_kind)
        self._lock = threading.Lock()
        self._token_alias_cache: dict[str, tuple[int, ...]] = {}

        logger.info(
            "Loading tokenizer/model %s device=%s dtype=%s",
            model_id,
            self.device_kind,
            self.torch_dtype,
        )

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        added_pad = self._configure_tokenizer()

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        # transformers: torch_dtype deprecated in favour of dtype, but both work.
        load_kwargs["torch_dtype"] = self.torch_dtype
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        # device_map="auto" только на CUDA: на CPU/MPS accelerate уводит веса на disk/meta и ломает forward.
        if self.device_kind == "cuda" and device_map is not None:
            load_kwargs["device_map"] = device_map
        elif self.device_kind == "cpu":
            load_kwargs["low_cpu_mem_usage"] = False

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:  # noqa: BLE001 — пробрасываем как EngineError
            raise EngineError(f"failed to load model {model_id!r}: {exc}") from exc

        if self.device_kind == "mps":
            self.model.to("mps")
        elif self.device_kind == "cpu":
            self.model.to("cpu")

        if added_pad:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.input_device = self._infer_input_device()
        logger.info("Model ready on %s (input device %s)", self.device_kind, self.input_device)

    def _configure_tokenizer(self) -> bool:
        """Left pad/truncate. Возвращает True, если в словарь добавлен новый pad-токен."""
        tok = self.tokenizer
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token_id is not None:
            return False
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            return False
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
            return False
        tok.add_special_tokens({"pad_token": "[PAD]"})
        return True

    def _infer_input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise EngineError("model has no parameters") from exc

    # --- public API ---------------------------------------------------------

    @property
    def device_label(self) -> str:
        return f"{self.device_kind}:{self.input_device}"

    def predict(
        self,
        state: str,
        questions: dict[str, Question],
        *,
        temperature: float | None = None,
    ) -> DecisionResponse:
        """Прогнать все вопросы к ``state`` одним forward-pass и вернуть типизированные ответы."""
        if not questions:
            raise EngineError("questions must be non-empty")
        temp = float(self.temperature if temperature is None else temperature)
        if temp <= 0:
            raise EngineError("temperature must be > 0")

        prepared = [self._prepare(state, key, question) for key, question in questions.items()]
        prompts = [item.prompt for item in prepared]

        encode_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        if self.device_kind == "cuda":
            encode_kwargs["pad_to_multiple_of"] = 8
        encoded = self.tokenizer(prompts, **encode_kwargs)
        encoded = {name: tensor.to(self.input_device) for name, tensor in encoded.items()}

        with self._lock, torch.inference_mode():
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
            # clone: отпускаем полный тензор [B, T, V] до декодирования ответов
            last_logits = outputs.logits[:, -1, :].float().clone()
            del outputs

        answers: dict[str, Answer] = {}
        for row, item in enumerate(prepared):
            answers[item.key] = self._decode_row(last_logits[row], item, temp)
        return DecisionResponse(answers=answers)

    # --- internals ----------------------------------------------------------

    def _prepare(self, state: str, key: str, question: Question) -> _PreparedItem:
        prompts = render_batch(
            state,
            {key: question},
            tokenizer=self.tokenizer,
            prompt_format=self.prompt_format,
        )
        rows = option_rows(question)
        labels = tuple(label for label, _ in rows)
        alias_ids = tuple(self._letter_token_ids(LETTERS[i]) for i in range(len(rows)))
        for i, aliases in enumerate(alias_ids):
            if not aliases:
                raise EngineError(
                    f"tokenizer {self.model_id!r} produced no token id for option "
                    f"{LETTERS[i]!r} (question {key!r})"
                )
        return _PreparedItem(
            key=key,
            question=question,
            prompt=prompts[key],
            labels=labels,
            alias_ids=alias_ids,
        )

    def _letter_token_ids(self, letter: str) -> tuple[int, ...]:
        cached = self._token_alias_cache.get(letter)
        if cached is not None:
            return cached

        ids: list[int] = []
        seen: set[int] = set()
        for variant in (letter, f" {letter}", f"\n{letter}"):
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in seen:
                seen.add(encoded[0])
                ids.append(int(encoded[0]))
        if not ids:
            encoded = self.tokenizer.encode(letter, add_special_tokens=False)
            if encoded:
                ids.append(int(encoded[0]))
        aliases = tuple(ids)
        self._token_alias_cache[letter] = aliases
        return aliases

    def _option_logits(self, row_logits: torch.Tensor, item: _PreparedItem) -> torch.Tensor:
        """Максимум по алиасам токена буквы → логит варианта (без дробления массы)."""
        gathered: list[torch.Tensor] = []
        for aliases in item.alias_ids:
            index = torch.tensor(aliases, device=row_logits.device, dtype=torch.long)
            gathered.append(row_logits.index_select(0, index).max())
        return torch.stack(gathered)

    def _softmax(self, option_logits: torch.Tensor, temperature: float) -> torch.Tensor:
        scaled = option_logits / temperature
        return F.softmax(scaled, dim=-1)

    def _decode_row(self, row_logits: torch.Tensor, item: _PreparedItem, temperature: float) -> Answer:
        option_logits = self._option_logits(row_logits, item)
        probs = self._softmax(option_logits, temperature)
        values = [float(p) for p in probs.detach().cpu().tolist()]
        best = int(probs.argmax().item())

        question = item.question
        if isinstance(question, QuestionChoice):
            dist = {label: _round_prob(p) for label, p in zip(item.labels, values, strict=True)}
            return ChoiceResponse(
                choice=item.labels[best],
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
            p_true = values[0]  # A = true / yes
            return NoulResponse(
                noul=_round_prob(p_true),
                confidence=_round_prob(max(p_true, 1.0 - p_true)),
            )

        raise EngineError(f"unsupported question type {type(question)!r}")
