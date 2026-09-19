#!/usr/bin/env python3
"""Сравнение torch vs vLLM gateway на ``dataset.jsonl`` (разная конкурентность)."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from evaluate import DEFAULT_DATASET, load_dataset, metrics_summary, run_eval

DEFAULT_TORCH = "http://127.0.0.1:8000/v1/decisions"
DEFAULT_VLLM = "http://127.0.0.1:8000/v1/decisions"


def _parse_concurrency(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _fmt_pct(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.1%}"


def _fmt_ms(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.1f}"


def _fmt_rps(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.2f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Backend",
        "Conc",
        "SchemaViol",
        "Choice",
        "Score",
        "Noul",
        "Mean(ms)",
        "P50",
        "P95",
        "P99",
        "RPS",
        "HTTPerr",
    ]
    col_widths = [max(len(h), 8) for h in headers]

    def row_cells(r: dict[str, Any]) -> list[str]:
        return [
            str(r["backend"]),
            str(r["concurrency"]),
            _fmt_pct(r["schema_violation_rate"]),
            _fmt_pct(r["choice_top1"]),
            _fmt_pct(r["score_top1"]),
            _fmt_pct(r["noul_top1"]),
            _fmt_ms(r["latency_mean_ms"]),
            _fmt_ms(r["latency_p50_ms"]),
            _fmt_ms(r["latency_p95_ms"]),
            _fmt_ms(r["latency_p99_ms"]),
            _fmt_rps(r["throughput_rps"]),
            str(r["http_errors"]),
        ]

    for i, cells in enumerate([headers] + [row_cells(r) for r in rows]):
        line = " | ".join(c.ljust(col_widths[j]) for j, c in enumerate(cells))
        print(line)
        if i == 0:
            print("-+-".join("-" * w for w in col_widths))


async def probe_endpoint(endpoint: str, timeout: float) -> bool:
    base = endpoint.rsplit("/v1/decisions", 1)[0]
    health_url = f"{base}/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(health_url)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def run_matrix(
    jobs: list[tuple[str, str, int]],
    dataset: Path,
    timeout: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend, endpoint, concurrency in jobs:
        print(f"\n>>> {backend} concurrency={concurrency} endpoint={endpoint}")
        if not await probe_endpoint(endpoint, timeout=min(timeout, 15.0)):
            print(f"    SKIP: {endpoint} not reachable (/health failed)", file=sys.stderr)
            rows.append(
                {
                    "backend": backend,
                    "concurrency": concurrency,
                    "schema_violation_rate": float("nan"),
                    "choice_top1": float("nan"),
                    "score_top1": float("nan"),
                    "noul_top1": float("nan"),
                    "latency_mean_ms": float("nan"),
                    "latency_p50_ms": float("nan"),
                    "latency_p95_ms": float("nan"),
                    "latency_p99_ms": float("nan"),
                    "throughput_rps": float("nan"),
                    "http_errors": -1,
                    "requests": 0,
                }
            )
            continue

        t0 = time.perf_counter()
        metrics = await run_eval(endpoint, dataset, concurrency, timeout)
        elapsed = time.perf_counter() - t0
        rows.append(
            metrics_summary(
                metrics,
                backend=backend,
                concurrency=concurrency,
                elapsed_s=elapsed,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch vs vLLM OpenJev backends on the benchmark dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--torch-endpoint",
        default=DEFAULT_TORCH,
        help="Decisions API URL for torch backend (default: port 8000).",
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=DEFAULT_VLLM,
        help="Decisions API URL for vLLM gateway (often same port after compose).",
    )
    parser.add_argument(
        "--concurrency",
        default="1,4,8",
        help="Comma-separated concurrency levels to test.",
    )
    parser.add_argument(
        "--backends",
        default="torch,vllm",
        help="Comma-separated backends to run (torch, vllm).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    levels = _parse_concurrency(args.concurrency)
    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]

    jobs: list[tuple[str, str, int]] = []
    for backend in backends:
        endpoint = args.torch_endpoint if backend == "torch" else args.vllm_endpoint
        if backend not in ("torch", "vllm"):
            print(f"Unknown backend {backend!r}, skip", file=sys.stderr)
            continue
        for c in levels:
            jobs.append((backend, endpoint, c))

    print("OpenJev backend comparison")
    print(f"Dataset: {args.dataset} ({len(load_dataset(args.dataset))} cases)")

    rows = asyncio.run(run_matrix(jobs, args.dataset, args.timeout))
    print("\n" + "=" * 72)
    print_table(rows)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
