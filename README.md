# Anima Benches

Curated benchmark harnesses and public benchmark reports for
[Anima](https://github.com/Arizona-dev/Anima).

This repository is intentionally benchmark-focused. It does not ship the Anima
server source tree itself; instead, it contains:

- LoCoMo evaluation harnesses and public reports
- LongMemEval evaluation harnesses
- Docker and compose helpers for running benchmarks against an Anima server

## Best Public Scores

### LoCoMo

| Run | Scope | Score | Source |
| --- | --- | --- | --- |
| Best focused run | `conv-26` only, 152 questions | **93.4%** | `locomo/RESULTS.md` |
| Best broad run | all 10 conversations, 1,540 questions | **86.0%** | `locomo/REPORT.md` |
| Best broad LLM judge score | all 10 conversations | **0.864** | `locomo/REPORT.md` |

LongMemEval harnesses are included here, but this repo does not yet publish a
curated headline LongMemEval score.

## Repository Layout

- `locomo/`: LoCoMo harnesses, configs, Docker helpers, and curated reports
- `longmemeval/`: LongMemEval harnesses, configs, and Docker helpers
- `Dockerfile`: combined benchmark runner image
- `docker-compose.yml`: combined benchmark stack for LoCoMo + LongMemEval

## Quickstart

### Direct runner

Start an Anima server separately, then run a benchmark harness against it.

```bash
cd locomo
python run.py \
  --anima-url http://localhost:3000 \
  --llm-model qwen/qwen3-32b \
  --judge-model llama-3.3-70b-versatile \
  --search-limit 100 \
  --output results/locomo-run.json
```

### Docker workflow

The compose files in this repo assume you already built an `anima-server`
container image from the main Anima repository.

```bash
docker build --platform linux/amd64 -t bench-all .
docker compose up
```

## Datasets

Benchmark datasets are not bundled in this repo. The harnesses auto-download
their upstream datasets when the requested local path does not exist. See
[DATASETS.md](DATASETS.md).

## License

This repository is released under the Apache License 2.0. See
[LICENSE](LICENSE).
