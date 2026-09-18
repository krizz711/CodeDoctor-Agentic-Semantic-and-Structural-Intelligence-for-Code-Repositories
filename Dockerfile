FROM python:3.11-slim

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt requirements-eval.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-eval.txt
RUN pip install --no-cache-dir faiss-cpu fastapi uvicorn pyyaml

# Copy full application codebase
COPY . .

# Build C++ product and demo targets
RUN make all || true

# Expose backend API and developer UI
EXPOSE 8000

# Default command: start the CodeDoctor service
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
