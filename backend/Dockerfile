FROM python:3.13-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[serve,speech]"
RUN mkdir -p /models \
    && python -m diagnostic_model.sequential_ebm \
       data/generated/synthetic_scenarios.csv --output-dir /models
ENV EBM_LIVE_MODEL=/models/ebm_sequential_models.joblib \
    STOPPING_MODEL=/models/ebm_stopping_controller.joblib \
    CONSULTATION_DB=/app/runtime/consultations.sqlite3

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
