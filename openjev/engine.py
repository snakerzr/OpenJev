"""DecisionEngine: batched vLLM ``/v1/completions`` + logprob slicing (System One)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from openjev.decoding import option_logprobs_from_top, pack_answer, softmax_from_logprobs
from openjev.errors import EngineError
from openjev.prompter import option_rows, render_batch
from openjev.schemas import Answer, DecisionResponse, Question

logger = logging.getLogger(__name__)

# vLLM 0.8.x: жёсткий потолок logprobs (см. BadRequestError max allowed: 20).
DEFAULT_LOGPROBS = 20


@dataclass(frozen=True, slots=True)
class _PreparedItem:
    key: str
    question: Question
    labels: tuple[str, ...]


class DecisionEngine:
    """System One через внешний vLLM OpenAI API (PagedAttention, ``max_tokens=1``)."""

    def __init__(
        self,
        model_id: str,
        vllm_url: str = "http://localhost:8001/v1",
        *,
        temperature: float = 1.0,
        timeout: float = 120.0,
        logprobs: int = DEFAULT_LOGPROBS,
        prompt_format: str = "instruct",
        max_length: int = 4096,
        trust_remote_code: bool = False,
    ) -> None:
        if temperature <= 0:
            raise EngineError("temperature must be > 0")
        if logprobs < 1:
            raise EngineError("logprobs must be >= 1")
        if max_length < 2:
            raise EngineError("max_length must be >= 2")

        self.model_id = model_id
        self.model_name = model_id
        self.base_url = vllm_url.rstrip("/")
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.logprobs = int(logprobs)
        self.prompt_format = prompt_format
        self.max_length = int(max_length)
        self.truncate_prompt_tokens = self.max_length - 1
        self.trust_remote_code = trust_remote_code

        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"failed to load tokenizer for {model_id!r}: {exc}") from exc

        self.device_label = f"vllm:{self.base_url}"
        self.dtype_label = "gateway"

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DecisionEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        """Проверка ``GET /v1/models`` на стороне vLLM."""
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"vLLM health check failed: {exc}") from exc
        models = body.get("data", [])
        ids = [m.get("id") for m in models if isinstance(m, dict)]
        ready = any(mid == self.model_id for mid in ids) if ids else bool(models)
        return {"ready": ready, "models": ids, "base_url": self.base_url}

    def predict(
        self,
        state: str,
        questions: dict[str, Question],
        *,
        temperature: float | None = None,
    ) -> DecisionResponse:
        if not questions:
            raise EngineError("questions must be non-empty")

        temp = float(self.temperature if temperature is None else temperature)
        keys = list(questions.keys())
        prompts_map = render_batch(
            state,
            questions,
            tokenizer=self.tokenizer,
            prompt_format=self.prompt_format,
        )
        prepared = [
            _PreparedItem(
                key=key,
                question=questions[key],
                labels=tuple(label for label, _ in option_rows(questions[key])),
            )
            for key in keys
        ]
        prompts = [prompts_map[item.key] for item in prepared]

        payload = {
            "model": self.model_id,
            "prompt": prompts,
            "max_tokens": 1,
            "logprobs": self.logprobs,
            "temperature": temp,
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
        }

        try:
            resp = self._client.post("/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"vLLM completions request failed: {exc}") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != len(prepared):
            raise EngineError(
                f"vLLM returned {len(choices) if isinstance(choices, list) else 0} choices, "
                f"expected {len(prepared)}"
            )

        answers: dict[str, Answer] = {}
        for item, choice in zip(prepared, choices, strict=True):
            answers[item.key] = self._decode_choice(item, choice, temp)
        return DecisionResponse(answers=answers)

    def _decode_choice(self, item: _PreparedItem, choice: Any, temperature: float) -> Answer:
        if not isinstance(choice, dict):
            raise EngineError("invalid vLLM choice object")

        logprobs_block = choice.get("logprobs")
        if not isinstance(logprobs_block, dict):
            raise EngineError("missing logprobs in vLLM response")

        top_list = logprobs_block.get("top_logprobs")
        if not isinstance(top_list, list) or not top_list:
            raise EngineError("missing top_logprobs in vLLM response")

        top0 = top_list[0]
        if not isinstance(top0, dict):
            raise EngineError("invalid top_logprobs[0]")

        top_map: dict[str, float] = {}
        for tok, lp in top0.items():
            try:
                top_map[str(tok)] = float(lp)
            except (TypeError, ValueError):
                continue

        num_options = len(item.labels)
        slice_lps = option_logprobs_from_top(top_map, num_options)
        probs = softmax_from_logprobs(slice_lps, temperature)
        return pack_answer(item.question, item.labels, probs)
