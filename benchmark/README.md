# OpenJev Decisions API Benchmark

Набор эталонных сценариев и скрипт оценки для локального **System One** inference engine (`POST /v1/decisions`).

## Структура

| Файл | Назначение |
|------|------------|
| `dataset.jsonl` | Тест-кейсы: `request` (валидное тело API) + `ground_truth` |
| `generate_data.py` | Генерация/сброс датасета и добавление синтетических кейсов |
| `evaluate.py` | Прогон через эндпоинт и отчёт по метрикам |

## Установка

Из **корня** репозитория (зависимость `httpx` — группа `benchmark` в `pyproject.toml`):

```bash
uv sync --group benchmark
```

Дополнительные пакеты не требуются: валидация ответа выполняется в `evaluate.py` по контракту из `ARCHITECTURE.md`.

## Подготовка датасета

Сгенерировать или пересоздать `dataset.jsonl` (38+ сценариев по умолчанию):

```bash
uv run python benchmark/generate_data.py --reset
```

Добавить синтетические routing-кейсы к существующему файлу:

```bash
uv run python benchmark/generate_data.py --extend 10
```

## Запуск оценки

1. Запустите Decisions API (по умолчанию `http://localhost:8000`), например:

   ```bash
   uv run openjev --model Qwen/Qwen2.5-1.5B-Instruct
   ```

   vLLM gateway:

   ```bash
   docker compose -f docker-compose.vllm.yml up --build
   # или
   uv run openjev --backend vllm --vllm-url http://127.0.0.1:8001/v1
   ```

2. Выполните:

```bash
uv run python benchmark/evaluate.py --endpoint http://localhost:8000/v1/decisions
```

### Сравнение torch vs vLLM

```bash
uv run python benchmark/run_comparison.py --concurrency 1,4,8
```

Параметры `--torch-endpoint` и `--vllm-endpoint` — если бэкенды на разных портах.

Параметры:

- `--dataset path/to/dataset.jsonl` — альтернативный датасет
- `--concurrency 8` — параллельные запросы
- `--timeout 120` — таймаут одного запроса (секунды)

## Метрики в отчёте

1. **Choice Top-1 Accuracy** — доля ответов, где `choice` входит в `expected_labels`.
2. **Score Top-1 Match** — совпадение `score` с `target_score` или `acceptable_scores`.
3. **Noul** — точность по порогу 0.5, доля ответов с `noul` в `prob_range`, **Brier score**, **ECE** (10 бинов).
4. **Schema Violation Rate** — доля ответов, не прошедших проверку типов/полей (цель: **0%**).
5. **Latency** — P50 / P95 / P99 и среднее время запроса (мс).

## Категории в датасете

- `support_routing` — маршрутизация тикетов (`choice`)
- `sentiment_score` — рубрика раздражения/срочности (`score`)
- `guardrails_safety` — injection, безопасность SQL, семантические вопросы (`noul`)
- `edge_cases` — короткие/длинные тексты, опечатки, инвариантность порядка `criteria`
- `ambiguity` — калибровка: ожидаемые вероятности в «серой зоне» (~35–65%)

## Формат `ground_truth`

Для каждого ключа в `request.questions`:

**choice**

```json
"route": { "type": "choice", "expected_labels": ["billing"] }
```

**score**

```json
"urgency": { "type": "score", "target_score": 2, "acceptable_scores": [2, 3] }
```

**noul**

```json
"check": {
  "type": "noul",
  "expected_bool": true,
  "prob_range": [0.8, 1.0]
}
```

Для спорных кейсов используйте узкий диапазон вокруг 0.5, например `[0.35, 0.65]`.
