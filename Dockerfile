# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY documents ./documents
COPY README.md .

ENV PYTHONPATH=/app

CMD ["python", "app/main.py"]