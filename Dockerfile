# Dockerfile for ManualIQ (FastAPI and/or Gradio)
FROM python:3.11-slim

# Install system dependencies (required for docling/opencv/etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed docs logs

# Expose the default port
EXPOSE 7860

# Command to run the server
CMD ["python", "scripts/app.py"]
