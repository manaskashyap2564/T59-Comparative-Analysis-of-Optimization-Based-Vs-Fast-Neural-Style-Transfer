FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements pehle (cache ke liye)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# backend code copy karo
COPY backend/ ./backend/

# src/ folder bhi copy karo — NST modules yahan hain
COPY src/ ./src/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

WORKDIR /app/backend
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]