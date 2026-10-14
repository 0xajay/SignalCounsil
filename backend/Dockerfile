FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables for production python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend source code
COPY . /app

# Start FastAPI application listening dynamically on Cloud Run's $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
