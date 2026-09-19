"""CLI-запуск OpenJev Decisions API (``python -m openjev`` / ``python run.py``)."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from openjev.server import DEFAULT_MODEL_ID, DEFAULT_VLLM_URL, create_app

PRODUCTION_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenJev Decisions API — System One inference server (0 output tokens).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL_ID),
        help=(
            f"Hugging Face model id (default: {DEFAULT_MODEL_ID}; "
            f"production: {PRODUCTION_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--backend",
        "--engine-type",
        dest="backend",
        choices=["torch", "vllm"],
        default=os.environ.get("OPENJEV_BACKEND", "torch"),
        help="Inference backend: local PyTorch or vLLM OpenAI API gateway.",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("OPENJEV_VLLM_URL", DEFAULT_VLLM_URL),
        help=f"vLLM OpenAI base URL (default: {DEFAULT_VLLM_URL}).",
    )
    parser.add_argument("--host", default=os.environ.get("OPENJEV_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENJEV_PORT", "8000")))
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("OPENJEV_TEMPERATURE", "1.0")),
        help="Softmax temperature for candidate-token calibration (default: 1.0).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.environ.get("OPENJEV_MAX_LENGTH", "4096")),
        help="Left-truncation limit for the tokenized prompt (torch backend).",
    )
    parser.add_argument(
        "--prompt-format",
        choices=["auto", "chatml", "instruct"],
        default=os.environ.get("OPENJEV_PROMPT_FORMAT", "auto"),
        help="Chat wrapper: auto-detect tokenizer chat_template, ChatML, or Instruct.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("OPENJEV_DTYPE", "auto"),
        help="Weight dtype: auto | bfloat16 | float16 | float32 (torch backend).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers (torch backend).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENJEV_LOG_LEVEL", "info"),
        help="Uvicorn / application log level.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload (dev only; do not use with large model loads).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(
        model_id=args.model,
        backend=args.backend,
        vllm_url=args.vllm_url,
        temperature=args.temperature,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=str(args.log_level).lower(),
        reload=args.reload,
    )
