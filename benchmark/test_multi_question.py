#!/usr/bin/env python3
"""Latency scaling: one shared ``state``, 1 vs 3 vs 8 questions (prefix caching)."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import httpx

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/decisions"

# ~500+ tokens of support context (logs + metadata) for prefix reuse tests.
LONG_STATE = """Support ticket #88421 — Enterprise customer Acme Corp (plan: Pro, seats: 240).
Opened: 2026-03-14 09:12 UTC by ops@acme.example | Region: eu-west-1 | SLA: 4h P1

SUMMARY: After billing plan downgrade from Enterprise to Pro, invoice PDF export fails silently.
Users report blank PDFs; CSV export still works. Issue started exactly after subscription change.

--- CONVERSATION ---
[09:12] Customer: We downgraded yesterday. Today finance cannot export March invoices as PDF.
[09:18] Agent: Can you confirm browser and whether CSV works?
[09:22] Customer: Chrome 122, CSV fine, PDF spinner then empty file (0 bytes).
[09:35] Agent: Escalated to billing eng. Transaction IDs: TX-99102, TX-99103.
[10:01] Customer: Also seeing duplicate line item for seat count 240 vs active 198.

--- APPLICATION LOG (redacted) ---
2026-03-14T09:40:11Z WARN billing.subscription plan_change from=enterprise to=pro org_id=acme-442
2026-03-14T09:40:12Z INFO invoice.render format=pdf invoice_ids=[4401,4402,4403] worker=pdf-v3
2026-03-14T09:40:13Z ERROR invoice.render pdf_bytes=0 template_id=ent_legacy org_plan=pro mismatch
2026-03-14T09:40:13Z ERROR invoice.render fallback_failed reason=template_not_found
2026-03-14T09:41:02Z INFO export.csv rows=1842 status=ok
2026-03-14T09:55:00Z WARN seat_reconcile licensed=240 active=198 delta=42 billing_cycle=2026-03-01

--- INTERNAL NOTES ---
- Possible entitlements cache stale after plan change.
- PDF template still references enterprise branding asset path.
- Finance deadline: month-end close in 36 hours (customer flagged urgent).
- Prior ticket #88100 similar (resolved by template remap).

--- ATTACHMENTS ---
export_attempt.pdf (0 bytes), invoice_march_partial.csv (checksum ok), screenshot_blank.pdf.png

Customer sentiment: frustrated but cooperative. Requests ETA before EOD EU.
"""


def _choice_q(key: str, instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _score_q(key: str, instructions: str, criteria: list[str]) -> dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def _noul_q(instructions: str) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {
            "true": "Condition clearly holds given the ticket",
            "false": "Condition does not hold",
        },
    }


def build_scenarios() -> list[tuple[str, dict[str, Any]]]:
    route_criteria = {
        "billing": "Payments, invoices, subscriptions, plan changes, refunds",
        "bugs": "Application errors, broken exports, crashes, regressions",
        "feature": "Product requests and roadmap feedback",
        "spam": "Irrelevant or abusive content",
    }
    urgency = ["low", "medium", "high", "critical"]
    sentiment = ["calm", "neutral", "frustrated", "angry"]

    eight = {
        "route": _choice_q("route", "Route this ticket to the correct team.", route_criteria),
        "urgent": _noul_q("Does the customer convey time pressure before month-end close?"),
        "frustration": _score_q(
            "frustration",
            "Rate customer frustration level.",
            sentiment,
        ),
        "pdf_bug": _noul_q("Is the PDF export failure likely a product bug rather than user error?"),
        "billing_plan": _noul_q("Is the issue plausibly caused by the recent plan downgrade?"),
        "duplicate_seats": _noul_q("Does the ticket mention a seat count billing discrepancy?"),
        "escalation": _score_q(
            "escalation",
            "How urgently should engineering escalate?",
            urgency,
        ),
        "spam_check": _noul_q("Is this message spam or phishing?"),
    }

    three = {
        "route": eight["route"],
        "frustration": eight["frustration"],
        "urgent": eight["urgent"],
    }

    one = {"urgent": eight["urgent"]}

    return [
        ("1 question (noul)", one),
        ("3 questions (choice+score+noul)", three),
        ("8 questions (mixed)", eight),
    ]


def post_decisions(
    client: httpx.Client,
    endpoint: str,
    state: str,
    questions: dict[str, Any],
    *,
    repeats: int,
) -> list[float]:
    latencies_ms: list[float] = []
    payload = {"state": state, "questions": questions}
    for _ in range(repeats):
        t0 = time.perf_counter()
        resp = client.post(endpoint, json=payload, timeout=120.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        if "answers" not in body or len(body.get("answers", {})) != len(questions):
            raise RuntimeError(f"unexpected response keys: {list(body.keys())}")
        latencies_ms.append(elapsed_ms)
    return latencies_ms


def summarize(latencies_ms: list[float]) -> str:
    if len(latencies_ms) == 1:
        return f"{latencies_ms[0]:.1f} ms"
    return (
        f"p50={statistics.median(latencies_ms):.1f} ms, "
        f"mean={statistics.mean(latencies_ms):.1f} ms, "
        f"min={min(latencies_ms):.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-question latency on one shared state")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--repeats", type=int, default=3, help="Timed requests per scenario (after warmup).")
    parser.add_argument("--warmup", action="store_true", default=True)
    args = parser.parse_args()

    print("OpenJev multi-question latency (shared state prefix)")
    print(f"Endpoint: {args.endpoint}")
    print(f"State length (chars): {len(LONG_STATE)}\n")

    scenarios = build_scenarios()
    results: list[tuple[str, int, list[float]]] = []

    with httpx.Client() as client:
        if args.warmup:
            try:
                post_decisions(
                    client,
                    args.endpoint,
                    LONG_STATE,
                    scenarios[0][1],
                    repeats=1,
                )
                print("Warmup OK\n")
            except httpx.HTTPError as exc:
                print(f"Warmup failed ({exc}). Is the gateway running on :8000?\n")
                raise SystemExit(1) from exc

        for label, questions in scenarios:
            n_q = len(questions)
            try:
                lat = post_decisions(
                    client,
                    args.endpoint,
                    LONG_STATE,
                    questions,
                    repeats=args.repeats,
                )
            except httpx.HTTPError as exc:
                print(f"[FAIL] {label}: {exc}")
                raise SystemExit(1) from exc
            results.append((label, n_q, lat))
            print(f"  {label}: {n_q} questions → {summarize(lat)}")

    if len(results) >= 3:
        lat1 = statistics.median(results[0][2])
        lat8 = statistics.median(results[2][2])
        ratio = lat8 / lat1 if lat1 > 0 else float("inf")
        linear_ratio = 8.0
        print()
        print(f"8Q / 1Q latency ratio (p50): {ratio:.2f}x (linear would be ~{linear_ratio:.0f}x)")
        if ratio < linear_ratio * 0.85:
            print("OK: sub-linear scaling suggests shared prefill / batching (APC + vLLM).")
        else:
            print(
                "Note: ratio is high — check vLLM --enable-prefix-caching, "
                "gateway batching, or GPU load."
            )


if __name__ == "__main__":
    main()
