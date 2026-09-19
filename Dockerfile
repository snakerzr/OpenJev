# Лёгкий OpenJev gateway (vLLM backend — без PyTorch в образе).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "httpx>=0.27.0" \
    "pydantic>=2.9.0" \
    "transformers>=4.45.0"

COPY openjev /app/openjev
COPY pyproject.toml README.md /app/

ENV PYTHONPATH=/app \
    OPENJEV_VLLM_URL=http://vllm:8000/v1 \
    OPENJEV_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    OPENJEV_PROMPT_FORMAT=instruct \
    OPENJEV_MAX_LENGTH=1024

EXPOSE 8000

CMD ["uvicorn", "openjev.server:app", "--host", "0.0.0.0", "--port", "8000"]
