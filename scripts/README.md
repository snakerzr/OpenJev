# Дистилляция Jev → локальный Qwen (QLoRA)

Три пути данных:

1. **Контрастные пары** (Nimble) — `generate_synthetic_samples.py` + **hard labels** → CE в `train_distill.py`.
2. **Teacher** (OpenRouter `typesafe/jev-1.13`) — `distill_collect.py` → **KL**.
3. **Готовые веса** — `jaredpalmer/kev-0.5b` / `kev-4b` через `OPENJEV_MODEL` (см. `docs/MODELS_SURVEY.md`).

## Окружение

Из корня репозитория (PyTorch только в optional extra, gateway остаётся лёгким):

```bash
uv sync --extra train
```

Требуется `OPENROUTER_API_KEY` для сбора. QLoRA 4-bit (`bitsandbytes`) — **Linux/WSL**; на Windows без bnb используйте `--no-4bit` или WSL2.

## 0. Проверка OpenRouter (Jev teacher)

```bash
$env:OPENROUTER_API_KEY="..."
uv run python scripts/probe_openrouter.py --also-latest
```

## 1. Сырые сценарии

Из бенчмарка:

```bash
uv run python scripts/prepare_raw_samples.py
```

Синтетика контрастными парами (OpenAI / совместимый шлюз):

```bash
uv sync --extra synthetic
$env:OPENAI_API_KEY="..."   # или только OPENROUTER_API_KEY (через openrouter.ai/api/v1)
uv run python scripts/generate_synthetic_samples.py --pairs-count 50 --concurrency 5
```

По умолчанию **50 пар = 100 строк** в `scripts/raw_samples.jsonl` (два `state` на пару + `hard_labels`). Формат: `{"state", "questions", "hard_labels"?}` — как Decisions API.

## 2. Сбор teacher soft labels

```bash
$env:OPENROUTER_API_KEY="sk-or-..."
uv run python scripts/distill_collect.py --input scripts/raw_samples.jsonl --output scripts/distill_dataset.jsonl --concurrency 10
```

## 3. LoRA-обучение

```bash
uv run python scripts/train_distill.py --data scripts/distill_dataset.jsonl --output-dir ./distilled_model --epochs 3 --batch-size 1 --grad-accum 4 --lr 5e-5 --loss-mode auto --brier-weight 0.1
```

Промпт по умолчанию **`instruct`**. **`--loss-mode auto`**: строки с `teacher_answers` → KL; с `hard_labels` (без teacher) → cross-entropy. LR по умолчанию **5e-5** (Kev: 2e-4 часто портит базу).

## 4. vLLM с дистиллированными весами

Пример override (не меняет дефолтный compose):

```yaml
# docker-compose.distill.yml
services:
  vllm:
    volumes:
      - ${USERPROFILE}/.cache/huggingface:/root/.cache/huggingface
      - ./distilled_model:/app/model
    command: >
      --model /app/model
      --port 8000
      --gpu-memory-utilization 0.85
      --max-model-len 1024
      --dtype bfloat16
      --enforce-eager
```

```bash
docker compose -f docker-compose.yml -f docker-compose.distill.yml up --build
```

Gateway (`openjev-gateway`) без изменений — укажите тот же путь модели в `OPENJEV_MODEL`, если токенизатор лежит в `./distilled_model`.
