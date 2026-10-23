FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY connect4/ connect4/
COPY web/ web/
COPY configs/ configs/
COPY scripts/ scripts/
COPY README.md pyproject.toml .

RUN mkdir -p checkpoints/pool data && \
    python3 scripts/train_fast.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
