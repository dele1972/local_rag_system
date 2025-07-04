# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for unstructured and Office documents
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    curl \
    netcat-traditional \
    dnsutils \
    tesseract-ocr \
    tesseract-ocr-deu \
    libtesseract-dev \
    libreoffice \
    # Zusätzliche Abhängigkeiten für HTML-Sanitization
    libxml2-dev \
    libxslt-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY documents ./documents
COPY README.md .

ENV PYTHONPATH=/app
# Tesseract Umgebungsvariable setzen
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Add a healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

CMD ["python", "app/main.py"]