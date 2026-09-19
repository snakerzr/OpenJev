"""CLI-запуск OpenJev Decisions API (``python -m openjev`` / ``python run.py``)."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from openjev.server import DEFAULT_MODEL_ID, DEFAULT_VLLM_URL, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenJev Decisions API — vLLM gateway (System One, 0 output tokens).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENJEV_MODEL", DEFAULT_MODEL_ID),
        help=f"Hugging Face model id (tokenizer + vLLM model name, default: {DEFAULT_MODEL_ID}).",
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
        help="Max prompt tokens; vLLM uses truncate_prompt_tokens=max_length-1.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=["auto", "chatml", "instruct"],
        default=os.environ.get("OPENJEV_PROMPT_FORMAT", "instruct"),
        help=(
            "Prompt wrapper: instruct (recommended for System One), chatml, or auto."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("OPENJEV_VLLM_TIMEOUT", "120")),
        help="HTTP timeout for vLLM requests (seconds).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPENJEV_LOG_LEVEL", "info"),
        help="Uvicorn / application log level.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload (dev only).",
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
        vllm_url=args.vllm_url,
        temperature=args.temperature,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
        trust_remote_code=args.trust_remote_code,
        timeout=args.timeout,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=str(args.log_level).lower(),
        reload=args.reload,
    )
