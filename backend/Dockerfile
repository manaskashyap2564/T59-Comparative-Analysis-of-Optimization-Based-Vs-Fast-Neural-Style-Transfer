FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — Cloud Run pe GPU nahi, 2GB bacha lo
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# requirements copy karo (root se, backend/ prefix nahi)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# sirf backend code copy karo
COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]