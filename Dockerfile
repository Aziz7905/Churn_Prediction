FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8001 5002

CMD ["sh", "-c", "mkdir -p artifacts/mlruns artifacts/models artifacts/processed artifacts/reports && uvicorn serve_api:app --host 0.0.0.0 --port 8000 & uvicorn serve_web:app --host 0.0.0.0 --port 8001 & mlflow server --backend-store-uri sqlite:///artifacts/mlflow.db --default-artifact-root file:///app/artifacts/mlruns --host 0.0.0.0 --port 5002 & wait"]
