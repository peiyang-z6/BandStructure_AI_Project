FROM python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
LABEL org.opencontainers.image.title="BandStructure MCP v2" \
      org.opencontainers.image.source="https://github.com/peiyang-z6/BandStructure_AI_Project" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
    BAND_MCP_UPLOAD_ISOLATION=1 BAND_MCP_STORE_DIR=/data/artifacts \
    BAND_MCP_TOKEN_FILE=/data/private/token
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 band && useradd -m -u 10001 -g band band \
    && mkdir -p /data /app && chown band:band /data
WORKDIR /app
COPY docker/requirements.txt docker/requirements.txt
RUN python -m pip install --no-cache-dir -r docker/requirements.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -m pip check
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"
ENTRYPOINT ["bandstructure-mcp"]
CMD ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
