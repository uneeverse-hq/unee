# Unee server: llama.cpp (CPU) + the Unee API (Jev-compatible /v1/systemone, OpenAI-compatible chat), one container.
#   docker build -t unee .
#   docker run -p 8000:8000 -v /path/to/models:/models -e MODEL=/models/unee-0.8b-Q4_K_M.gguf unee
FROM ghcr.io/ggml-org/llama.cpp:server
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY unee ./unee
RUN pip3 install --no-cache-dir --break-system-packages .
ENV MODEL=/models/unee-0.8b-Q4_K_M.gguf THREADS=4 SLOTS=4 CTX=16384 TEMPERATURE=1.0
EXPOSE 8000
ENTRYPOINT []
CMD unee serve --model "$MODEL" --llama-server /app/llama-server --host 0.0.0.0 --port 8000 \
    --threads "$THREADS" --slots "$SLOTS" --ctx "$CTX" --temperature "$TEMPERATURE"
