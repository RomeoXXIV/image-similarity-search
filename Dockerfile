FROM python:3.11-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/static/uploads /app/static/features /app/static/results /app/data && \
    mkdir -p /home/appuser && \
    chown -R appuser:appuser /app /home/appuser && \
    chmod -R 755 /app /home/appuser && \
    chmod -R 777 /app/static/results /app/static/uploads

USER appuser

EXPOSE 8080

ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Configuration de Matplotlib pour utiliser un backend non-interactif
ENV MPLBACKEND=Agg

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "4", \
     "--worker-class", "sync", \
     "--worker-connections", "1000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--timeout", "60", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--preload", \
     "app.main:app"]
