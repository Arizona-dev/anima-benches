# Benchmark runner image
# Build from benchmarks/locomo/:
#   docker build -t bench-runner .

FROM python:3.12-slim

WORKDIR /bench

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py config.a100.toml ./

ENTRYPOINT ["python", "run.py"]
