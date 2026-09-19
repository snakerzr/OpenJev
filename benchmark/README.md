# OpenJev Decisions API Benchmark

Набор сценариев и скрипт оценки для `POST /v1/decisions` (шлюз + vLLM).

## Установка

```bash
uv sync --group benchmark
```

## Запуск

1. Поднимите стек: `docker compose up --build` или `uv run openjev --vllm-url ...`
2. Прогон:

```bash
uv run python benchmark/evaluate.py --endpoint http://localhost:8000/v1/decisions
```

Параметры: `--dataset`, `--concurrency`, `--timeout`.

## Метрики

Choice / Score / Noul accuracy, schema violations, latency P50/P95/P99.

## Датасет

`dataset.jsonl` — 38 кейсов; пересоздание: `uv run python benchmark/generate_data.py --reset`.
