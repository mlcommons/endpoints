# syntax=docker/dockerfile:1

## -----------------------------------------------------
FROM dhi.io/python:3.14-debian13-sfw-dev AS build-stage


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

RUN --mount=type=secret,id=HF_TOKEN,dst=/run/secrets/hf_token export HF_TOKEN=$(cat /run/secrets/hf_token)

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

RUN python /opt/LiveCodeBench_Datasets/generate.py \
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

COPY --from=build-stage --chmod=0555 /app/venv /app/venv
COPY --from=build-stage --chmod=0555 /opt/LiveCodeBench_Datasets /opt/LiveCodeBench_Datasets
COPY lcb_serve.py /app/lib/lcb_serve.py
COPY run_lcb_tests.py /app/lib/run_lcb_tests.py
COPY generate.py /app/lib/generate.py
COPY _server.py /app/server.py

# Make lcb_serve.py available as a module
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Launch the WebSocket server with long-running connection support
# Default port 13835
# - timeout-keep-alive: Allow connections to stay open for hours
# - ws-ping-interval: Send ping every 30s to keep connection alive
# - ws-ping-timeout: Wait 10s for pong response before considering connection dead
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "13835", \
     "--timeout-keep-alive", "7200", \
     "--ws-ping-interval", "30", \
     "--ws-ping-timeout", "10"]
