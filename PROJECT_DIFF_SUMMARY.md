# Project Diff Summary

**SOURCE:** `Cloud_Project_Final/Cloud_Project`  
**TARGET:** `Evolutionary-Vision-Optimization-Platform`  
**Analysis Date:** 2026-05-16

---

## Architecture Comparison

| Component | SOURCE | TARGET (before) | TARGET (after) |
|-----------|--------|-----------------|----------------|
| CNN Model | `model/` (cnn, train, evaluate, run_baseline) | Missing | Added |
| API | FastAPI v2.0 + MLOps routes | FastAPI v1.0, predict only | Merged v2.0 |
| Experiments | Full `experiments/` module | Missing | Added |
| Docker | Dockerfile + Compose + UI Dockerfile | Dockerfile only | Full suite |
| Security | Hardcoded credentials | None | `.env` pattern |
| Tests | `tests/test_loaders.py` (old signature) | `tests/test_loaders.py` (old signature) | Fixed signature |
| UI | Streamlit dashboard + Dockerfile | Streamlit dashboard only | Added Dockerfile |
| Docs | Cheatsheet + Deployment guide (with credentials) | Minimal README | Sanitized docs |

---

## File-by-File Diff

### `api/main.py`
- **SOURCE (247 lines):** v2.0 — ExperimentManager, CORS, 7 routes, DB prediction logging
- **TARGET before (112 lines):** v1.0 — just /health + /predict, MIME validation, FileNotFoundError
- **TARGET after (175 lines):** Merged best of both — v2.0 feature set + v1.0 error handling quality
- **Key additions:** `ExperimentConfig`/`ResumeConfig` models, `/start_experiment`, `/resume_experiment`, `/get_logs/{id}`, `/experiment_status/{run_name}`, CORS middleware

### `data/loaders.py`
- **SOURCE:** `get_loaders(dataset: str = "cifar10", batch_size: int = 64)`
- **TARGET before:** `get_loaders(batch_size: int)` — MISSING dataset parameter
- **TARGET after:** Fixed to match SOURCE signature with default values
- **Impact:** Critical — without this fix, `model/train.py` crashes with TypeError

### `ui/dashboard/utils.py`
- **SOURCE (147 lines):** Hardcoded EC2 IP `http://13.51.70.11:8000/predict`, broken exception order
- **TARGET before (164 lines):** Uses `os.environ.get("API_URL")` but simpler `load_results`
- **TARGET after (163 lines):** Best of both — env var config, proper exception order, robust CSV loading

### `docker/Dockerfile`
- **SOURCE (15 lines):** Multi-module COPY, healthcheck, correct layer caching
- **TARGET before (9 lines):** `COPY . .` (copies everything), no healthcheck
- **TARGET after (21 lines):** Selective COPY for each module, healthcheck, curl installed

### `docker/requirements.txt` + root `requirements.txt`
- **SOURCE:** Added `sqlalchemy>=2.0.0`, `psycopg2-binary>=2.9.0`
- **TARGET before:** Missing DB libraries
- **TARGET after:** Added both

### `docker-compose.yml`
- **SOURCE only (32 lines):** Hardcoded `DATABASE_URL` in environment
- **TARGET after (32 lines):** Uses `${DATABASE_URL}` from `.env`, both services with healthchecks

### `.gitignore`
- **TARGET before:** Standard Python gitignore
- **TARGET after:** Added `.pem`, `*.key`, `*.pth`, `ga_checkpoint.json`

---

## New Modules Added

### `experiments/` — MLOps Experiment Tracking
```
experiments/
├── __init__.py          — package init
├── db_models.py         — SQLAlchemy ORM (Experiment, RunLog, PredictionLog, Checkpoint)
├── logger.py            — DBLogger class
├── checkpoint.py        — DBCheckpointManager class
└── manager.py           — ExperimentManager (run/resume with Spot interruption support)
```

**Capabilities:**
- Start/stop experiments with full DB logging
- Checkpoint GA population state to AWS RDS after each generation
- Resume interrupted experiments from last checkpoint (Spot instance resilience)
- Log every inference prediction to DB

### `model/` — CNN Training Pipeline
```
model/
├── __init__.py          — package init
├── cnn.py               — SimpleCNN (configurable conv blocks + FC classifier)
├── train.py             — Training loop (AdamW, CosineAnnealing, gradient clipping)
├── evaluate.py          — Test-set evaluation + single-image inference
├── run_baseline.py      — Baseline training script
└── checkpoints/         — Model checkpoints directory (binaries gitignored)
```

**Capabilities:**
- GA-tunable architecture (num_filters, num_layers, dropout, batch_size, lr)
- Kaiming/Xavier weight initialization
- Macro F1 without extra library dependencies
- Wrapped checkpoint format with config metadata

---

## Size Comparison

| Metric | SOURCE | TARGET (before) | TARGET (after) |
|--------|--------|-----------------|----------------|
| Python source files | 22 | 12 | 24 |
| Total source lines (est.) | ~4,200 | ~2,800 | ~5,100 |
| API version | 2.0.0 | 1.0.0 | 2.0.0 |
| Docker services | 2 (api + ui) | 1 (api) | 2 (api + ui) |
| MLOps routes | 4 | 0 | 4 |
| Critical bugs | 5 | 1 (loaders sig) | 0 |
| Hardcoded secrets | 4 | 1 (EC2 IP) | 0 |
