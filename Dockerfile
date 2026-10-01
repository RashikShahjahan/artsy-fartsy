FROM python:3.11.13-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc libcairo2-dev pkg-config
RUN pip wheel --no-cache-dir --wheel-dir=/wheels pycairo==1.28.0

FROM python:3.11.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libcairo2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels pycairo==1.28.0
COPY src/artsy_harness/artcanvas.py /renderer/artcanvas.py
ENV PYTHONPATH=/renderer PYTHONDONTWRITEBYTECODE=1
WORKDIR /output
USER 65534:65534
ENTRYPOINT ["python", "-B", "/input/program.py"]
