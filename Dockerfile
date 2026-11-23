# Build for linux/amd64 explicitly: Cloud Run runs x86, and building on Apple
# Silicon without this flag produces an arm64 image whose XGBoost fits may not
# reproduce the cloud numbers.
#
#   docker build --platform linux/amd64 --build-arg PYTHON_VERSION=3.12 -t drift-prototype .
#
# Set PYTHON_VERSION to the major.minor reported by scripts/print_local_versions.py
# in the Notebook 05 kernel. Do not guess it.
ARG PYTHON_VERSION=3.12
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV PROTOTYPE_CONFIG=/app/config/prototype_config.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY sql/ ./sql/
COPY config/ ./config/

ENTRYPOINT ["python", "src/main.py"]
