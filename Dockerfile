FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip3 install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* README.md ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-interaction --no-ansi --no-root

# Copy source code
COPY enhanced_text_normalization.py /app/

# Copy model repository (only the models, no tokenizer)
COPY text_normalization_triton/model_repository /models

# Create necessary directories
RUN mkdir -p /app/models_cache /app/data/dictionary

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--allow-http=true", "--allow-grpc=true", "--allow-metrics=true"]