# PiFi Dockerfile (CUDA 11.8)
# Base on PyTorch runtime to avoid re-downloading CUDA wheels
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    WANDB_MODE=offline

WORKDIR /app

# System deps (git for HF, make for convenience)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates make && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user that matches typical host UID/GID 1000 for devcontainers
# This fixes VS Code Server install and HOME directory permissions when running as 1000:1000
RUN groupadd -g 1000 dev || true \
 && id -u dev >/dev/null 2>&1 || useradd -m -u 1000 -g 1000 -s /bin/bash dev \
 && mkdir -p /opt/hf-cache /app/cache /app/preprocessed /app/models /app/checkpoints /app/results /app/tensorboard_logs \
 && chown -R dev:dev /home/dev /opt/hf-cache /app

# Copy and install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Create cache/artifact directories with write permissions
RUN mkdir -p /opt/hf-cache /app/cache /app/preprocessed /app/models /app/checkpoints /app/results /app/tensorboard_logs && \
    chmod -R 777 /opt/hf-cache /app/cache /app/preprocessed /app/models /app/checkpoints /app/results /app/tensorboard_logs && \
    chown -R dev:dev /app

# Ensure project is importable and set default root
ENV PYTHONPATH=/app \
    ROOT_DIR=/app

# Entrypoint: run first-run ILM classification, then exec CMD
COPY docker/entrypoint.sh /usr/local/bin/pifi-entrypoint
RUN chmod +x /usr/local/bin/pifi-entrypoint
ENTRYPOINT ["/usr/local/bin/pifi-entrypoint"]
CMD ["bash", "-lc", "tail -f /dev/null"]
