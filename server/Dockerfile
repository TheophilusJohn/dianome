# dianome/server (Phase 6 self-host): the split-inference server half in one image.
#
#   docker build -t dianome/server .                          # CUDA (default): pytorch/pytorch runtime base
#   docker build --build-arg VARIANT=cpu -t dianome/server:cpu .   # CPU / Apple-silicon Docker (no MPS inside a container)
#   docker run -e SPLIT_SIGNING_KEY=… -e MODEL=qwen2.5-0.5b-instruct -p 8765:8765 -v dianome-models:/models dianome/server
#
# Env: MODEL (default qwen2.5-0.5b-instruct), SPLIT_SIGNING_KEY (verifies the Worker's session tokens; share the value
# with the Worker), SPLIT_TOKEN (optional static bearer for direct clients), SERVE_ARGS (extra `serve` flags, e.g.
# --no-microbench), HF_TOKEN (only for gated repos). Weights download from Hugging Face on first start into /models
# (mount a volume there) and the health check is GET /plan, which is public. Nothing else is exposed.
ARG VARIANT=cuda

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base-cuda

FROM python:3.12-slim AS base-cpu
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

FROM base-${VARIANT} AS server
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL=qwen2.5-0.5b-instruct \
    SERVE_ARGS=
WORKDIR /app
COPY pyproject.toml README.md ./
COPY dianome_server ./dianome_server
COPY bench ./bench
COPY probes ./probes
RUN pip install --no-cache-dir . && mkdir -p /models
VOLUME ["/models"]
EXPOSE 8765
# The model download can take minutes on first start: a long start period before the health check counts failures.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/plan', timeout=5).status == 200 else 1)"
ENTRYPOINT ["sh", "-c", "exec dianome-server serve --model \"$MODEL\" --host 0.0.0.0 --port 8765 $SERVE_ARGS"]
