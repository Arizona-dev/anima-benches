# Combined benchmark runner — LoCoMo + LongMemEval
# Build from benchmarks/:
#   docker build --platform linux/amd64 -t bench-all .

FROM python:3.12-slim

WORKDIR /bench

# Install deps (both benchmarks share the same requirements)
COPY locomo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# LoCoMo benchmark
COPY locomo/run.py locomo/config.a100.toml locomo/requirements.txt locomo/
COPY locomo/REPORT.md locomo/RESULTS.md locomo/compare_servers.py locomo/

# LongMemEval benchmark
COPY longmemeval/run.py longmemeval/config.a100.toml longmemeval/requirements.txt longmemeval/

# Runner script
COPY run_all.sh .
RUN chmod +x run_all.sh

ENTRYPOINT ["bash", "run_all.sh"]
