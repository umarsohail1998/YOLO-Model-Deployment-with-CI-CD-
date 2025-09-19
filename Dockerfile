# Use slim image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Optional: Pre-download YOLO weights during build to speed startup
# This will download yolov8n.pt to the working directory
RUN python - <<'PY'
from ultralytics import YOLO
try:
    YOLO("yolov8n.pt")
    print("Downloaded yolov8n.pt")
except Exception as e:
    print("Could not pre-download weights:", e)
PY

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
