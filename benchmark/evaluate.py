#!/usr/bin/env python3
"""Run benchmark cases against a local Decisions API and report metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


@dataclass
class QuestionGT:
    qtype: str
    expected_labels: list[str] | None = None
    target_score: int | None = None
    acceptable_scores: list[int] | None = None
    expected_bool: bool | None = None
    prob_range: tuple[float, float] | None = None


@dataclass
class EvalMetrics:
    choice_total: int = 0
    choice_correct: int = 0
    score_total: int = 0
    score_correct: int = 0
    noul_total: int = 0
    noul_bool_correct: int = 0
    noul_prob_in_range: int = 0
    brier_sum: float = 0.0
    schema_errors: int = 0
    request_errors: int = 0
    total_requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    noul_preds: list[tuple[float, int]] = field(default_factory=list)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_ground_truth(gt: dict[str, Any]) -> dict[str, QuestionGT]:
    out: dict[str, QuestionGT] = {}
    for qkey, spec in gt.items():
        pr = spec.get("prob_range")
        out[qkey] = QuestionGT(
            qtype=spec["type"],
            expected_labels=spec.get("expected_labels"),
            target_score=spec.get("target_score"),
            acceptable_scores=spec.get("acceptable_scores"),
            expected_bool=spec.get("expected_bool"),
            prob_range=(float(pr[0]), float(pr[1])) if pr else None,
        )
    return out


def validate_answer(qtype: str, answer: Any) -> str | None:
    """Return error message if schema invalid, else None."""
    if not isinstance(answer, dict):
        return "answer is not an object"
    if answer.get("type") != qtype:
        return f"type mismatch: expected {qtype}, got {answer.get('type')}"
    if qtype == "noul":
        n = answer.get("noul")
        if not isinstance(n, (int, float)) or not (0.0 <= float(n) <= 1.0):
            return "invalid noul probability"
        conf = answer.get("confidence")
        if conf is not None and not isinstance(conf, (int, float)):
            return "invalid confidence"
    elif qtype == "choice":
        if not isinstance(answer.get("choice"), str):
            return "missing choice string"
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            return "invalid probabilities dict"
        try:
            s = sum(float(v) for v in probs.values())
        except (TypeError, ValueError):
            return "non-numeric probabilities"
        if not math.isfinite(s) or s <= 0:
            return "probabilities sum invalid"
    elif qtype == "score":
        sc = answer.get("score")
        if not isinstance(sc, int):
            return "score must be int"
        probs = answer.get("probabilities")
        if not isinstance(probs, list) or not probs:
            return "invalid score probabilities list"
        try:
            vals = [float(p) for p in probs]
        except (TypeError, ValueError):
            return "non-numeric score probabilities"
        if sc < 0 or sc >= len(vals):
            return "score index out of range"
    else:
        return f"unknown question type {qtype}"
    return None


def compute_ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> float:
    if not pairs:
        return float("nan")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    ece = 0.0
    n = len(pairs)
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(p for p, _ in bucket) / len(bucket)
        avg_acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - avg_acc)
    return ece


async def evaluate_one(
    client: httpx.AsyncClient,
    endpoint: str,
    case: dict[str, Any],
    metrics: EvalMetrics,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        payload = case["request"]
        gt_map = parse_ground_truth(case["ground_truth"])
        t0 = time.perf_counter()
        try:
            resp = await client.post(endpoint, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics.latencies_ms.append(latency_ms)
            metrics.total_requests += 1
            if resp.status_code != 200:
                metrics.request_errors += 1
                return
            try:
                body = resp.json()
            except json.JSONDecodeError:
                metrics.schema_errors += 1
                return
            answers = body.get("answers")
            if not isinstance(answers, dict):
                metrics.schema_errors += 1
                return
            for qkey, q_gt in gt_map.items():
                if qkey not in answers:
                    metrics.schema_errors += 1
                    continue
                ans = answers[qkey]
                err = validate_answer(q_gt.qtype, ans)
                if err:
                    metrics.schema_errors += 1
                    continue
                if q_gt.qtype == "choice":
                    metrics.choice_total += 1
                    label = ans["choice"]
                    if q_gt.expected_labels and label in q_gt.expected_labels:
                        metrics.choice_correct += 1
                elif q_gt.qtype == "score":
                    metrics.score_total += 1
                    sc = int(ans["score"])
                    ok = False
                    if q_gt.acceptable_scores:
                        ok = sc in q_gt.acceptable_scores
                    elif q_gt.target_score is not None:
                        ok = sc == q_gt.target_score
                    if ok:
                        metrics.score_correct += 1
                elif q_gt.qtype == "noul":
                    metrics.noul_total += 1
                    p = float(ans["noul"])
                    y = 1 if q_gt.expected_bool else 0
                    metrics.brier_sum += (p - y) ** 2
                    metrics.noul_preds.append((p, y))
                    pred_bool = p >= 0.5
                    if q_gt.expected_bool is not None and pred_bool == q_gt.expected_bool:
                        metrics.noul_bool_correct += 1
                    if q_gt.prob_range:
                        lo, hi = q_gt.prob_range
                        if lo <= p <= hi:
                            metrics.noul_prob_in_range += 1
        except httpx.HTTPError:
            metrics.request_errors += 1


async def run_eval(
    endpoint: str,
    dataset_path: Path,
    concurrency: int,
    timeout: float,
) -> EvalMetrics:
    cases = load_dataset(dataset_path)
    metrics = EvalMetrics()
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [evaluate_one(client, endpoint, c, metrics, sem) for c in cases]
        await asyncio.gather(*tasks)
    return metrics


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def metrics_summary(
    metrics: EvalMetrics,
    *,
    backend: str = "unknown",
    concurrency: int = 1,
    elapsed_s: float | None = None,
) -> dict[str, Any]:
    """Сводка метрик для таблиц сравнения (``run_comparison.py``)."""
    total_q = metrics.choice_total + metrics.score_total + metrics.noul_total
    viol_rate = (metrics.schema_errors / total_q) if total_q else float("nan")

    choice_acc = (
        metrics.choice_correct / metrics.choice_total if metrics.choice_total else float("nan")
    )
    score_acc = metrics.score_correct / metrics.score_total if metrics.score_total else float("nan")
    noul_acc = metrics.noul_bool_correct / metrics.noul_total if metrics.noul_total else float("nan")

    mean_lat = statistics.mean(metrics.latencies_ms) if metrics.latencies_ms else float("nan")
    p50 = percentile(metrics.latencies_ms, 50) if metrics.latencies_ms else float("nan")
    p95 = percentile(metrics.latencies_ms, 95) if metrics.latencies_ms else float("nan")
    p99 = percentile(metrics.latencies_ms, 99) if metrics.latencies_ms else float("nan")

    rps = float("nan")
    if elapsed_s is not None and elapsed_s > 0 and metrics.total_requests:
        rps = metrics.total_requests / elapsed_s

    return {
        "backend": backend,
        "concurrency": concurrency,
        "schema_violation_rate": viol_rate,
        "choice_top1": choice_acc,
        "score_top1": score_acc,
        "noul_top1": noul_acc,
        "latency_mean_ms": mean_lat,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "throughput_rps": rps,
        "http_errors": metrics.request_errors,
        "requests": metrics.total_requests,
    }


def print_report(metrics: EvalMetrics) -> None:
    print("=" * 60)
    print("OpenJev Decisions API — Benchmark Report")
    print("=" * 60)

    if metrics.choice_total:
        acc = metrics.choice_correct / metrics.choice_total
        print(f"Choice Top-1 Accuracy: {acc:.3%} ({metrics.choice_correct}/{metrics.choice_total})")
    else:
        print("Choice Top-1 Accuracy: n/a (no choice questions)")

    if metrics.score_total:
        acc = metrics.score_correct / metrics.score_total
        print(f"Score Top-1 Match:     {acc:.3%} ({metrics.score_correct}/{metrics.score_total})")
    else:
        print("Score Top-1 Match:     n/a (no score questions)")

    if metrics.noul_total:
        brier = metrics.brier_sum / metrics.noul_total
        ece = compute_ece(metrics.noul_preds)
        bool_acc = metrics.noul_bool_correct / metrics.noul_total
        calib = metrics.noul_prob_in_range / metrics.noul_total
        print(f"Noul Bool Accuracy:    {bool_acc:.3%} ({metrics.noul_bool_correct}/{metrics.noul_total})")
        print(f"Noul Prob-in-Range:    {calib:.3%} ({metrics.noul_prob_in_range}/{metrics.noul_total})")
        print(f"Noul Brier Score:      {brier:.4f} (lower is better)")
        print(f"Noul ECE (10-bin):     {ece:.4f} (lower is better)")
    else:
        print("Noul metrics:          n/a")

    total_q = metrics.choice_total + metrics.score_total + metrics.noul_total
    if total_q:
        viol_rate = metrics.schema_errors / total_q
        print(f"Schema Violation Rate: {viol_rate:.3%} ({metrics.schema_errors}/{total_q} answers)")
    else:
        print("Schema Violation Rate: n/a")

    if metrics.request_errors:
        print(f"HTTP / transport errors: {metrics.request_errors} requests")

    if metrics.latencies_ms:
        print(
            f"Latency (ms): P50={percentile(metrics.latencies_ms, 50):.1f}  "
            f"P95={percentile(metrics.latencies_ms, 95):.1f}  "
            f"P99={percentile(metrics.latencies_ms, 99):.1f}  "
            f"mean={statistics.mean(metrics.latencies_ms):.1f}"
        )
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Decisions API against benchmark dataset")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/decisions",
        help="Full URL to POST /v1/decisions",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to dataset.jsonl",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel requests")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        print("Run: python benchmark/generate_data.py --reset", file=sys.stderr)
        return 1

    metrics = asyncio.run(
        run_eval(args.endpoint, args.dataset, args.concurrency, args.timeout)
    )
    print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
