FROM python:3.12-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY backend/pyproject.toml backend/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy backend code
COPY backend /app/backend

# Set python path and command
ENV PYTHONPATH=/app
CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
