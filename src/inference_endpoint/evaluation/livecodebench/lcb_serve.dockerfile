# syntax=docker/dockerfile:1

## -----------------------------------------------------
FROM dhi.io/python:3.14-debian13-sfw-dev AS build-stage


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

RUN python -m venv /app/venv
RUN sfw pip install --no-cache-dir \
    datasets==3.6.0 \
    pandas==2.3.3 \
    fastapi==0.128.0 \
    uvicorn[standard]==0.40.0 \
    pydantic==2.12.5 \
    tqdm==4.67.1

RUN mkdir -p /opt/LiveCodeBench_Datasets/release_v6

COPY generate.py /opt/LiveCodeBench_Datasets/generate.py

RUN --mount=type=secret,id=HF_TOKEN,dst=/run/secrets/hf_token \
    export HF_TOKEN=$(cat /run/secrets/hf_token) \
    && python /opt/LiveCodeBench_Datasets/generate.py \
        --datasets-dir /opt/LiveCodeBench_Datasets \
        --variant release_v6
RUN chmod 444 -R /opt/LiveCodeBench_Datasets/*
RUN chmod 555 /opt/LiveCodeBench_Datasets

## -----------------------------------------------------
FROM dhi.io/python:3.14-debian13 AS runtime-stage

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

# enroot/pyxis starts containers through /bin/sh (switchroot); nv-sflow's srun
# operator wraps every task in `bash -c` (hardcoded in srun.py). This distroless
# runtime base has neither, so add a static busybox as /bin/sh, then create
# /bin/bash as a thin sh-wrapper via RUN (valid once /bin/sh exists above).
# busybox:*-musl is multi-arch so buildx --platform builds still resolve.
# NOTE: do NOT copy busybox as /bin/bash — busybox-musl omits the bash applet
# and fails with "bash: applet not found" when invoked under that name.
COPY --from=busybox:1.37.0-musl /bin/busybox /bin/sh
COPY --chmod=755 <<'EOF' /bin/bash
#!/bin/sh
exec /bin/sh "$@"
EOF

COPY --from=build-stage --chmod=0555 /app/venv /app/venv
COPY --from=build-stage --chmod=0555 /opt/LiveCodeBench_Datasets /opt/LiveCodeBench_Datasets
COPY lcb_serve.py /app/lib/lcb_serve.py
COPY run_lcb_tests.py /app/lib/run_lcb_tests.py
COPY generate.py /app/lib/generate.py
COPY _server.py /app/server.py

# Make lcb_serve.py available as a module
ENV PYTHONPATH="/app"

# Launch the WebSocket server with long-running connection support
# Default port 13835
# - timeout-keep-alive: Allow connections to stay open for hours
# - ws-ping-interval: Send ping every 30s to keep connection alive
# - ws-ping-timeout: Wait 10s for pong response before considering connection dead
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "13835", \
     "--timeout-keep-alive", "7200", \
     "--ws-ping-interval", "30", \
     "--ws-ping-timeout", "10"]
