# Unee on a free Hugging Face Space (Docker SDK, CPU Basic: 2 vCPU, 16 GB).
# Installs the published `unee` package and bakes the published 0.8B GGUF into the image at build time.
FROM ghcr.io/ggml-org/llama.cpp:server
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG UNEE_VERSION=""
RUN pip3 install --no-cache-dir --break-system-packages "unee${UNEE_VERSION:+==$UNEE_VERSION}"
ARG MODEL_URL=https://huggingface.co/uneeverse/unee-0.8b-GGUF/resolve/main/unee-0.8b-Q4_K_M.gguf
RUN mkdir -p /models && curl -fL --retry 5 -o /models/unee.gguf "$MODEL_URL" && chmod -R a+r /models
# Spaces run the container as user 1000 and route traffic to app_port 7860.
EXPOSE 7860
ENTRYPOINT []
CMD unee serve --model /models/unee.gguf --llama-server /app/llama-server --host 0.0.0.0 --port 7860 \
    --threads 2 --slots 2 --ctx 8192
