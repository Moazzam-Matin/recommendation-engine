FROM python:3.14-slim

WORKDIR /app

ENV MODEL_PATH=/app/artifacts/cf_model.joblib

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY pyproject.toml .
COPY src/ ./src/
COPY artifacts/ ./artifacts/

RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "recommendation_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

