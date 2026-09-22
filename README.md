# KONI-Forge Lite

A partial open release of **KONI-Forge**, the LLM development platform of the
Large-scale AI Research Center at the Korea Institute of Science and Technology
Information (KISTI). It accompanies the KONI-Forge system demonstration paper.

This edition keeps the core loop — *upload documents → measure what the model
already knows → fine-tune → chat* — driven by a small team of LLM agents. It
runs natively; the accelerator is detected at startup (CUDA, Apple Silicon MPS,
or CPU).

**Verified on:** macOS 26.6 (Apple M4, 32 GB, MPS), Python 3.13, Node 26,
torch 2.14, transformers 4.57, trl 1.6. The CUDA and CPU paths are implemented
and unit-tested but were not exercised end to end for this release.

## Pages

| Page | What it does |
|---|---|
| **Home** | Device memory and storage at a glance. |
| **Upload** | Source documents (PDF / DOCX / PPTX / HWPX / TXT / MD) indexed for RAG, or a QA dataset of `{question, answer}` pairs (JSON / JSONL) for training. |
| **Train** | Full fine-tuning or LoRA (TRL `SFTTrainer`) with live progress and loss charts. |
| **Chat** | Talk to a base or fine-tuned model, optionally grounded on the indexed documents. |

An agent panel runs alongside them: a supervisor LLM routes to **Retrieval**
(indexing and RAG status), **Boundary / KBD** (Knowledge Boundary Detection —
measures the base model's coverage and hallucination rate on your documents,
then recommends RAG, fine-tuning or a hybrid) and **Tuning** (recommends
hyperparameters, then launches the job). Every action is shown and confirmed
before it runs; set the autonomy to `auto` to skip the confirmation.

**Not in this edition:** QA data generation, preprocessing and augmentation,
benchmark evaluation, preference (DPO) training, distributed and multi-node
training, experiment tracking.

## Requirements

- Python 3.11+
- Redis (`brew install redis`, or `apt install redis-server`)
- Node 18+, to build the web UI
- An OpenAI API key — the agents and the KBD judge use it

A GPU is optional: training runs on CUDA, Apple Silicon (MPS) or CPU. Set
`KONI_DEVICE` to force one.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env          # set OPENAI_API_KEY and KONI_ADMIN_PW
npm --prefix UI ci
mkdir -p static && npm --prefix UI run build:deploy

./run.sh                      # Redis + API server + Celery worker
```

Open `http://localhost:8000` and sign in with the admin account from `.env`.
`./run.sh stop` stops the server and the worker.

### Getting a base model

Base models live under `storage/models/` in HuggingFace layout:

```bash
curl -X POST http://localhost:8000/api/models/download/hf \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model_id":"HuggingFaceTB/SmolLM2-135M-Instruct"}'
```

The download runs on the worker; poll `GET /api/models/download/{job_id}/status`.
Gated or private repos need `HF_TOKEN` in `.env`. `SmolLM2-135M-Instruct`
(260 MB) trains in seconds and is enough for a first run.

### Operator limits

`host.toml` caps model size, sequence length, batch size and job concurrency.
It is optional — copy `host.toml.example` to `./host.toml` to change the
defaults.

## Layout

```
api.py              FastAPI entrypoint
run.sh              start Redis, the API server and the worker
server/routers/     HTTP API
server/core/        pure helpers (device detection, paths, prompts, config)
modules/agents/     supervisor and the three specialists
modules/rag/        embeddings and vector store
celery_app/         worker tasks (indexing, model download, KBD probing, training)
pipelines/          inference and training-record helpers
UI/                 React + TypeScript front-end (Vite)
tests/              unit tests, plus opt-in e2e smoke tests
```

## Development

```bash
.venv/bin/python -m pytest tests/unit   # unit tests
.venv/bin/ruff check .                  # lint
npm --prefix UI run lint                # UI lint
npm --prefix UI run typecheck           # UI types
npm --prefix UI test                    # UI tests
```

End-to-end smoke tests run against a live stack:

```bash
KONI_E2E=1 KONI_E2E_USER=admin KONI_E2E_PASS=<pw> \
  .venv/bin/python -m pytest tests/e2e
```

## Notes

- **Celery pool.** macOS starts subprocesses with `spawn`, which Celery's
  prefork pool does not survive, so `run.sh` uses the thread pool there and
  prefork on Linux. Stopping a run is handled by the training callback, so it
  works under either pool.
- **First index is slow.** The first indexing job downloads the embedding model
  (about 1 GB); `INDEX_REQUEST_TIMEOUT_SECONDS` covers it.
- **Apple Silicon.** MPS has no bf16 autocast path, so training runs in fp32.

## License

Copyright © 2026 Korea Institute of Science and Technology Information (KISTI),
Large-scale AI Research Center.

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE)
(`PolyForm-Noncommercial-1.0.0`). Any noncommercial purpose is permitted,
including research, teaching and use by educational, public research and
government institutions. Commercial use requires a separate licence from KISTI.
