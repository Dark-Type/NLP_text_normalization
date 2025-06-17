FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /app

RUN pip3 install poetry

COPY pyproject.toml poetry.lock* README.md ./

RUN poetry config virtualenvs.create false

RUN poetry install --only=main --no-interaction --no-ansi --no-root


ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=""

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1", "--strict-model-config=false"]