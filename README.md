# OpenJev — Decisions API

Локальный **System One** inference-сервер: модель не генерирует текст (0 output tokens). Один forward-pass трансформера, срез логитов последнего токена, softmax только по разрешённым идентификаторам (`A`/`B`/`C` …). Ответ всегда строго типизирован: `choice`, `score` или `noul`.

```
f(State, {Questions}) → {Decisions}     P95: десятки–сотни мс на GPU
```

## Требования

- [uv](https://docs.astral.sh/uv/) (менеджер Python и зависимостей)
- Python 3.10+ (в репозитории зафиксирован **3.11** в `.python-version`)
- GPU (CUDA) рекомендуется; работают MPS (Apple) и CPU
- Hugging Face-модель instruct/chat (по умолчанию Qwen2.5)

## Установка

Из корня репозитория:

```bash
uv sync
```

С зависимостями для бенчмарка (`httpx`):

```bash
uv sync --group benchmark
```

Создаётся `.venv`, ставится пакет `openjev` в editable-режиме, версии фиксируются в `uv.lock`.

На **Windows/Linux** `torch` берётся с индекса PyTorch **CUDA 12.6** (`2.14.0+cu126` — актуальная stable CUDA-сборка на момент lock). На macOS — CPU/MPS wheel с PyPI.

Проверка GPU после установки:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Другой индекс CUDA (например `cu128`) — в `pyproject.toml` → `[tool.uv.index]` / `[tool.uv.sources]`, затем `uv lock && uv sync`.

## Запуск

Лёгкий прогон (1.5B):

```bash
uv run openjev --model Qwen/Qwen2.5-1.5B-Instruct
```

Продакшен-размер (7B):

```bash
uv run openjev --model Qwen/Qwen2.5-7B-Instruct
```

Эквиваленты:

```bash
uv run python run.py --model Qwen/Qwen2.5-1.5B-Instruct
uv run python -m openjev --model Qwen/Qwen2.5-1.5B-Instruct
```

Через uvicorn напрямую (модель из `OPENJEV_MODEL`):

```bash
# bash
OPENJEV_MODEL=Qwen/Qwen2.5-1.5B-Instruct uv run uvicorn openjev.server:app --host 0.0.0.0 --port 8000
```

```powershell
# PowerShell
$env:OPENJEV_MODEL="Qwen/Qwen2.5-1.5B-Instruct"; uv run uvicorn openjev.server:app --host 0.0.0.0 --port 8000
```

После старта: `GET http://127.0.0.1:8000/health` и схема — `http://127.0.0.1:8000/docs`.

### Полезные флаги CLI

| Флаг | Смысл |
|------|--------|
| `--temperature 1.0` | Температура softmax по кандидатам |
| `--max-length 4096` | Left-truncation промпта (обрезается начало `state`) |
| `--prompt-format auto` | `auto` / `chatml` / `instruct` |
| `--dtype auto` | `auto` · `bfloat16` · `float16` · `float32` |
| `--host` / `--port` | Адрес бинда (по умолчанию `0.0.0.0:8000`) |

### Обновление зависимостей

```bash
uv lock --upgrade-package transformers   # пример: обновить один пакет
uv sync
```

## Проверка curl

На Windows используйте `curl.exe` (не алиас PowerShell):

```powershell
curl.exe -s http://127.0.0.1:8000/v1/decisions `
  -H "Content-Type: application/json" `
  -d "@examples/request.json"
```

```bash
curl -s http://127.0.0.1:8000/v1/decisions \
  -H "Content-Type: application/json" \
  -d @examples/request.json
```

Заголовок ответа `X-Inference-Time-Ms` — время forward-pass в миллисекундах.

Неизвестный `type` вопроса → HTTP **422**.

## Контракт

| Тип | Вход `criteria` | Выход |
|-----|-----------------|--------|
| `choice` | словарь `{метка: описание}` | `choice` (argmax) + `probabilities` по меткам |
| `score` | список строк шкалы `0 .. N-1` | `score` — int argmax, `expected` — `Σ i·P_i`, вектор `probabilities` |
| `noul` | опционально `{true, false}` | `noul` = P(true) ∈ [0, 1] |

Все вопросы к одному `state` батчатся в один тензор `input_ids` (left-padding). Генерации нет: `use_cache=False`, читаются только `outputs.logits[:, -1, :]`.

## Бенчмарк

См. [`benchmark/README.md`](benchmark/README.md). После `uv sync --group benchmark`:

```bash
uv run python benchmark/evaluate.py --endpoint http://127.0.0.1:8000/v1/decisions
```

Сравнение **torch** vs **vLLM** (несколько уровней `--concurrency`):

```bash
# torch на :8000, vLLM gateway на :8000 после compose — запускайте по очереди
# или укажите разные порты:
uv run python benchmark/run_comparison.py \
  --torch-endpoint http://127.0.0.1:8002/v1/decisions \
  --vllm-endpoint http://127.0.0.1:8000/v1/decisions \
  --concurrency 1,4,8
```

### Результаты (референсный прогон)

**Модель:** `Qwen/Qwen2.5-1.5B-Instruct` (zero-shot, без дообучения).  
**Датасет:** `benchmark/dataset.jsonl` — **38** кейсов (`choice` / `score` / `noul`).  
**Железо:** NVIDIA GeForce RTX 3050 Laptop, **6 GB** VRAM, Windows + Docker Desktop (WSL2 backend для vLLM).

На одной 6 GB карте **не держите torch и vLLM одновременно** — сравнивайте последовательно (разные порты или остановка одного стека).

| Метрика | PyTorch (`--backend torch`, CUDA) | vLLM (Compose gateway, `c=4`) |
|--------|-----------------------------------|-------------------------------|
| Choice Top-1 | 80.0% (12/15) | **86.7%** (13/15) |
| Score Top-1 | **90.0%** (9/10) | 80.0% (8/10) |
| Noul (bool @ 0.5) | ~62% | **69.2%** (9/13) |
| Schema violations | **0%** | **0%** (38/38) |
| Latency P50 | ~183 ms | **~85 ms** |
| Latency P95 | ~1180 ms | ~628 ms |
| Latency mean | ~325 ms | ~154 ms |
| Concurrency | 2 | 4 |

**PyTorch:** `uv run openjev --backend torch --model Qwen/Qwen2.5-1.5B-Instruct --port 8002` (пример для сравнения), `torch 2.14.0+cu126`, `--max-length 4096`.

**vLLM:** `docker compose -f docker-compose.vllm.yml up --build` — образ `vllm/vllm-openai:v0.8.5`, `--max-model-len 1024`, `--gpu-memory-utilization 0.88`; gateway: `OPENJEV_MAX_LENGTH=1024`, усечение через `truncate_prompt_tokens = max_length - 1` (left-truncate длинных промптов, **38/38** без HTTP 400).

```bash
uv run python benchmark/evaluate.py --endpoint http://127.0.0.1:8000/v1/decisions --concurrency 4
```

Интерпретация: **P50 ~85 ms** попадает в типичный коридор System One / TypeSafe Jev (**70–500 ms**) для интерактивного синхронного HTTP. Рост **P95** на vLLM (~628 ms) — ожидаемая плата за prefill промптов ~1000 токенов вместо отказа по контексту; при **concurrency 4** хвост заметно ниже, чем у torch при **c=2** с сериализацией forward (~1.2 s P95).

## Running with vLLM (Docker Compose)

Стек: **vLLM** (GPU, порт **8001**) + **OpenJev gateway** (порт **8000**), оптимизирован под ~6 GB VRAM:

```bash
docker compose -f docker-compose.vllm.yml up --build
```

- Decisions API: http://localhost:8000 (`GET /health` → `backend: "vllm"`)
- vLLM OpenAI API напрямую: http://localhost:8001/v1

Бенчмарк против gateway:

```bash
uv run python benchmark/evaluate.py --endpoint http://127.0.0.1:8000/v1/decisions --concurrency 4
```

Локальный gateway без Docker (vLLM уже запущен на 8001):

```bash
uv run openjev --backend vllm --vllm-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-1.5B-Instruct
```

Переменные окружения: `OPENJEV_BACKEND=vllm`, `OPENJEV_VLLM_URL`, `OPENJEV_MODEL`, `OPENJEV_MAX_LENGTH` (должен совпадать с `--max-model-len` vLLM для корректного `truncate_prompt_tokens`).

## Архитектура

Подробности пайплайна, RLCD-loss и vLLM-вариант — в [`ARCHITECTURE.md`](ARCHITECTURE.md).
