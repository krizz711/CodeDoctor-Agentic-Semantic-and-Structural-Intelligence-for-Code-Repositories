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

# Install the submission recipe with an explicit Linux CPU wheel. Model weights
# and document caches are downloaded/built at runtime, outside the image build.
COPY requirements.txt requirements-submission.txt ./
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r requirements-submission.txt

# Copy full application codebase
COPY . .

# Build C++ product and demo targets
RUN make -B
RUN make -B test

# Expose backend API and developer UI
EXPOSE 8000

# Default command: start the CodeDoctor service
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
