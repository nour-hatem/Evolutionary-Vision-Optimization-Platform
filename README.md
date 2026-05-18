<div align="center">

# Evolutionary Vision Optimization Platform

**A cloud-native research platform that combines Genetic Algorithm-driven hyperparameter optimization with Convolutional Neural Networks to achieve state-of-the-art image classification on CIFAR-10 — deployed on AWS with full MLOps experiment tracking.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20RDS-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-RDS-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://aws.amazon.com/rds/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-AWS%20EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](http://13.48.139.52:8501/)

*Cloud Computing — Spring 2026*

</div>

---

## Table of Contents

- [Live Deployment](#live-deployment)
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Results](#results)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Genetic Algorithm + CNN Workflow](#genetic-algorithm--cnn-workflow)
- [Tech Stack](#tech-stack)
- [Cloud Infrastructure](#cloud-infrastructure)
- [API Endpoints](#api-endpoints)
- [Folder Structure](#folder-structure)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [Streamlit Dashboard](#streamlit-dashboard)
- [AWS Deployment](#aws-deployment)
- [Experiment Tracking](#experiment-tracking)
- [Evaluation](#evaluation)
- [Cost Analysis](#cost-analysis)
- [Screenshots](#screenshots)
- [Cloud Computing Bonus](#cloud-computing-bonus)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Live Deployment

The platform is fully deployed on **AWS EC2 (eu-north-1 / Stockholm)** and accessible right now — no local setup required.

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit Dashboard** | [http://13.48.139.52:8501/](http://13.48.139.52:8501/) | Interactive 4-page analytics UI |
| **FastAPI Swagger Docs** | [http://13.48.139.52:8000/docs](http://13.48.139.52:8000/docs) | Full REST API with live try-it-out |
| **API Health Check** | [http://13.48.139.52:8000/health](http://13.48.139.52:8000/health) | DB connectivity status (JSON) |
| **EC2 Instance** | `13.48.139.52` | AWS EC2 t3.micro — Ubuntu 22.04 — eu-north-1 |

### How to Access the System

#### Using the Dashboard (UI)

1. Open [http://13.48.139.52:8501/](http://13.48.139.52:8501/) in any browser.
2. Navigate using the sidebar:
   - **Home** — Platform overview, key metrics, optimization workflow.
   - **Prediction** — Upload any JPEG/PNG image; the system returns the CIFAR-10 class and a confidence score.
   - **Dashboard Analytics** — GA convergence chart, accuracy comparison, metrics radar chart.
   - **Experiment Logs** — Browse per-generation experiment data and evaluation results; export as CSV.

#### Using the API Directly

1. Open [http://13.48.139.52:8000/docs](http://13.48.139.52:8000/docs) — the interactive Swagger UI lists every endpoint.
2. Key endpoints to try:

```bash
# Health check
curl http://13.48.139.52:8000/health

# Classify an image (returns CIFAR-10 class + confidence)
curl -X POST http://13.48.139.52:8000/predict \
  -F "file=@your_image.jpg"

# Start a GA optimization experiment
curl -X POST http://13.48.139.52:8000/start_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "demo_run", "max_generations": 6, "population_size": 8, "mutation_rate": 0.3, "dataset": "cifar10", "interrupt_after": 3}'

# Check experiment status
curl http://13.48.139.52:8000/experiment_status/demo_run

# Resume an interrupted experiment
curl -X POST http://13.48.139.52:8000/resume_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "demo_run"}'
```

---

## Overview

The **Evolutionary Vision Optimization Platform** is an end-to-end cloud computing research project that integrates three tightly coupled components:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Evolutionary Optimizer** | DEAP + PyTorch | Genetic Algorithm that automatically evolves CNN hyperparameters |
| **Inference + MLOps API** | FastAPI + SQLAlchemy | REST API for image predictions and experiment tracking |
| **Monitoring Dashboard** | Streamlit + Plotly | Real-time visualization of experiments and predictions |

All three run as Docker containers on **AWS EC2**, with experiment data persisted to **AWS RDS PostgreSQL** — including full GA population checkpoint state so Spot-instance interruptions are fully recoverable without losing any training progress.

> **Project Proposal:** [`docs/proposal/Project_Idea_Proposal.pdf`](docs/proposal/Project_Idea_Proposal.pdf)
>
> **Interactive Experiment Diagram:** [`docs/diagrams/experiment_diagram.html`](docs/diagrams/experiment_diagram.html)
>
> **Cost Analysis Report:** [`docs/reports/cost_analysis.html`](docs/reports/cost_analysis.html)

---

## Problem Statement

### Problem 1 — Manual Hyperparameter Tuning

CNN performance is highly sensitive to architecture choices (number of layers, filter count) and training hyperparameters (learning rate, dropout, batch size). Manually searching this space requires dozens of expensive training runs with no principled strategy. This project replaces manual search with a **Genetic Algorithm** that treats each candidate configuration as a chromosome and evolves the population toward higher validation accuracy.

### Problem 2 — Cloud Training Interruption

Long ML training jobs on cloud infrastructure are vulnerable to **Spot instance reclamation** — the instance can be terminated mid-run with little notice, losing all in-progress computation. This platform addresses this with **per-generation checkpointing**: after each GA generation, the full population state is serialized and saved to AWS RDS PostgreSQL. If the instance is reclaimed, the run resumes from the last saved generation via a single API call.

---

## Results

The GA-optimized model significantly outperforms the hand-tuned baseline:

| Metric | Baseline CNN | GA-Optimized CNN | Improvement |
|--------|-------------|-----------------|-------------|
| **Test Accuracy** | 85.87% | **89.93%** | **+4.06%** |
| **F1 Macro** | 85.79% | **89.87%** | **+4.08%** |
| **F1 Weighted** | 85.79% | 89.87% | +4.08% |
| Inference (ms/batch) | 56.73 ms | 48.84 ms | −7.89 ms |
| Parameters | 3.4M | 12.2M | +8.9M |
| Model File Size | 13.53 MB | 49.04 MB | +35.51 MB |

> *Baseline results from `Evaluation/report.md`. Separate run baseline in `metrics.json` shows 86.54% accuracy — both are real results from different training seeds.*

### GA Convergence

The GA ran for **10 generations** with a population of **20 individuals**:

| Generation | Best Fitness | Avg Fitness |
|---|---|---|
| 0 | 70.02% | 56.97% |
| 5 | ~71.5% | ~63% |
| 10 | **73.38%** | **69.54%** |

The entire population improved (average fitness +12.57%), not just the single best individual — demonstrating effective evolutionary pressure.

### Best Configuration Found by GA

```json
{
  "lr": 0.0003,
  "num_filters": 96,
  "num_layers": 4,
  "dropout": 0.3,
  "batch_size": 32,
  "epochs": 15
}
```

*Source: `best_config.json`*

---

## Features

### Machine Learning

- **Genetic Algorithm** hyperparameter search over 5 dimensions (lr, num\_filters, num\_layers, dropout, batch\_size)
- **SimpleCNN** — configurable architecture with conv blocks, batch normalization, Dropout2d, and adaptive FC head
- **Fast-mode training** (3 epochs forced) during GA fitness evaluation for speed; full 15–20 epoch training for the final model
- **Cosine annealing** learning rate scheduler + gradient clipping (max\_norm=1.0) for stable convergence
- **Kaiming/Xavier weight initialization** for faster convergence from random init

### MLOps and Cloud

- **Spot-resilient checkpointing** — GA population state saved to AWS RDS after every generation; resume seamlessly after any interruption
- **Full experiment tracking** — every run, generation metric, and prediction logged to PostgreSQL across four tables
- **Start/resume experiments** via REST API without touching the server
- **Docker Compose** orchestration — single command brings up the entire platform

### API and Dashboard

- **Image classification API** — upload any JPEG/PNG image, get CIFAR-10 class + confidence score
- **MIME type validation** — rejects non-image uploads with HTTP 415
- **Experiment status API** — query any run's status, per-generation logs, and checkpoint in real time
- **Interactive Streamlit dashboard** — GA convergence charts, accuracy comparison, confidence distribution, metrics radar
- **CORS-enabled** — UI container calls API across the Docker bridge network
- **Healthcheck endpoints** — both containers expose health probes used by Docker and load balancers

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      AWS EC2 t3.micro                          │
│                    Ubuntu 22.04 | eu-north-1                   │
│                                                                │
│   ┌─────────────────────┐      ┌──────────────────────────┐   │
│   │   cifar-ui          │      │   cifar-api              │   │
│   │   Streamlit :8501   │─────▶│   FastAPI   :8000        │   │
│   │                     │      │                          │   │
│   │  • Home             │      │  GET  /                  │   │
│   │  • Prediction       │      │  GET  /health            │   │
│   │  • Analytics        │      │  POST /predict           │   │
│   │  • Experiment Logs  │      │  POST /start_experiment  │   │
│   └─────────────────────┘      │  POST /resume_experiment │   │
│                                │  GET  /get_logs/{id}     │   │
│                                │  GET  /experiment_status │   │
│                                └───────────┬──────────────┘   │
└────────────────────────────────────────────│────────────────── ┘
                                             │ SQLAlchemy / psycopg2
                                  ┌──────────▼────────────┐
                                  │   AWS RDS             │
                                  │   PostgreSQL          │
                                  │   db.t3.micro         │
                                  │                       │
                                  │  • experiments        │
                                  │  • run_logs           │
                                  │  • checkpoints        │
                                  │  • prediction_logs    │
                                  └───────────────────────┘
```

**Standalone GA Optimizer** (run separately, locally or on any machine):

```
GA module/ga_optimizer_final.py  (DEAP)
    │
    ├── Evolves 20 chromosomes × 10 generations
    ├── Fitness = model/train.py with fast_mode=True (3 epochs)
    ├── Saves ga_checkpoint.json after every generation
    ├── Writes best_config.json  → used for final full training
    └── Writes ga_log.csv        → displayed in dashboard
```

---

## Genetic Algorithm + CNN Workflow

### Step 1 — Chromosome Encoding

Each candidate configuration is encoded as a **5-gene integer chromosome** (indices into value lists):

```
Chromosome = [lr_idx, filter_idx, layer_idx, dropout_idx, batch_idx]

Search Space:
  lr_idx      → [0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001]   (7 values)
  filter_idx  → [32, 64, 96, 128]                                       (4 values)
  layer_idx   → [2, 3, 4, 5]                                            (4 values)
  dropout_idx → [0.3, 0.4, 0.5]                                         (3 values)
  batch_idx   → [32, 64, 128]                                            (3 values)

Total search space: 7 × 4 × 4 × 3 × 3 = 1,008 distinct configurations
```

### Step 2 — Fitness Evaluation

Each individual is evaluated by calling `model/train.py:train()` with `fast_mode=True` (3 epochs) and returning the validation accuracy as the fitness value.

### Step 3 — Evolution Loop

```
Generation 0:  Randomly initialize and evaluate all 20 individuals
For each generation 1 … N:
  1. Elitism:    carry best 1 individual unchanged to next generation
  2. Selection:  tournament selection (k=3) for remaining 19 slots
  3. Crossover:  two-point crossover, probability = 0.7 per pair
  4. Mutation:   random gene replacement — 0.3 prob per individual,
                 0.3 prob per gene when individual is selected
  5. Evaluate:   only re-evaluate individuals that changed (fitness cache)
  6. Checkpoint: save full population state to ga_checkpoint.json
```

**GA Parameters** (from `best_config.json`):

```json
{
  "population_size": 20,
  "n_generations": 10,
  "crossover_prob": 0.7,
  "mutation_prob": 0.4,
  "gene_mutpb": 0.35,
  "elite_size": 1,
  "seed": 42
}
```

### Step 4 — Spot-Resilient Checkpointing

```
Experiment interrupted at Generation N
          ↓
New instance / session starts
          ↓
python "GA module/ga_optimizer_final.py" --resume
          ↓
Loads ga_checkpoint.json → population state at Generation N
          ↓
Continues from Generation N+1
          ↓
Experiment completes as if never interrupted
```

The same pattern applies to the API-driven experiment manager: `POST /resume_experiment` loads the last checkpoint from RDS and resumes the loop.

### Step 5 — Final Training

The winning chromosome is decoded and the CNN is retrained for **15 full epochs** (no fast-mode) to produce the final `optimized_model.pth`.

### Step 6 — Reproducibility

All random seeds are fixed to `SEED = 42` at run start: Python `random`, NumPy, PyTorch CPU and CUDA, with `cudnn.deterministic = True`. The same seed always produces the same population and evolution path.

---

## Tech Stack

### Backend

| Library | Version | Role |
|---------|---------|------|
| **FastAPI** | ≥ 0.111 | REST API framework |
| **Uvicorn** | ≥ 0.29 | ASGI server |
| **PyTorch** | ≥ 2.2 | CNN training and inference |
| **Torchvision** | ≥ 0.17 | CIFAR-10 datasets and transforms |
| **Pillow** | ≥ 10.3 | Image decoding for inference |
| **SQLAlchemy** | ≥ 2.0 | ORM for PostgreSQL |
| **psycopg2-binary** | ≥ 2.9 | PostgreSQL driver |

### Evolutionary Optimizer

| Library | Version | Role |
|---------|---------|------|
| **DEAP** | latest | Genetic Algorithm framework (tournament selection, crossover, mutation) |
| **NumPy** | ≥ 1.24 | Fitness statistics |

> **Note:** DEAP is required for `GA module/ga_optimizer_final.py`. Install with `pip install deap` in the GA environment.

### Frontend

| Library | Version | Role |
|---------|---------|------|
| **Streamlit** | ≥ 1.32 | Multi-page dashboard |
| **Plotly** | ≥ 5.18 | Interactive charts (convergence, comparison, radar) |
| **Pandas** | ≥ 2.0 | Data manipulation and CSV loading |
| **Requests** | ≥ 2.31 | API communication |

### Infrastructure

| Service | Purpose |
|---------|---------|
| **AWS EC2 t3.micro** | Hosts API + UI Docker containers |
| **AWS RDS db.t3.micro** | Managed PostgreSQL for experiment tracking |
| **Docker + Docker Compose** | Container build and orchestration |
| **DockerHub** | Container image registry |

---

## Cloud Infrastructure

Deployed on **AWS EU-North-1 (Stockholm)**:

```
                    Internet
                        │
              ┌─────────▼──────────┐
              │   EC2 Security     │
              │   Group            │
              │   TCP 8000 (API)   │
              │   TCP 8501 (UI)    │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────────────┐
              │   EC2 t3.micro             │
              │   Ubuntu 22.04             │
              │                            │
              │  ┌──────────────────────┐  │
              │  │ cifar-api  :8000     │──┼──▶ RDS PostgreSQL
              │  │ restart:unless-stop  │  │    db.t3.micro :5432
              │  └──────────────────────┘  │    (experiments DB)
              │  ┌──────────────────────┐  │
              │  │ cifar-ui   :8501     │  │
              │  │ restart:unless-stop  │  │
              │  └──────────────────────┘  │
              └────────────────────────────┘
```

Both containers use `restart: unless-stopped` for automatic recovery after instance reboots. Both expose Docker healthchecks (curl probe, 30s interval, 3 retries).

---

## API Endpoints

**Live Base URL:** `http://13.48.139.52:8000`
**Interactive docs:** [http://13.48.139.52:8000/docs](http://13.48.139.52:8000/docs)

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service metadata and version |
| `GET` | `/health` | Liveness probe with DB connection status |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Upload a JPEG/PNG image → predicted CIFAR-10 class + confidence score |

**Request** (multipart/form-data):
```
file: <image file>   # JPEG, PNG, BMP, WEBP accepted
```

**Response:**
```json
{
  "prediction": "airplane",
  "confidence": 0.9342
}
```

Invalid (non-image) uploads return **HTTP 415 Unsupported Media Type**.

### MLOps — Experiment Tracking

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/start_experiment` | Start a new experiment run (runs as background task) |
| `POST` | `/resume_experiment` | Resume an interrupted experiment from its last RDS checkpoint |
| `GET` | `/get_logs/{experiment_id}` | Fetch per-generation metrics for a running or completed experiment |
| `GET` | `/experiment_status/{run_name}` | Full status: config, timings, logs, and checkpoint info |

**Start experiment request body:**
```json
{
  "run_name": "run_v1",
  "max_generations": 10,
  "population_size": 20,
  "mutation_rate": 0.3,
  "dataset": "cifar10",
  "interrupt_after": 3
}
```

Setting `interrupt_after: 3` simulates a Spot interruption after 3 generations — useful for demonstrating the checkpoint/resume workflow.

---

## Folder Structure

```
Evolutionary-Vision-Optimization-Platform/
│
├── README.md                         ← This file
├── requirements.txt                  ← Python dependencies (API + model)
├── docker-compose.yml                ← Multi-service orchestration
├── .env.example                      ← Environment variable template
├── .gitignore
├── best_config.json                  ← GA-discovered best hyperparameters
├── metrics.json                      ← Baseline training metrics (20 epochs)
├── check_db.py                       ← AWS RDS connection verifier
├── LICENSE                           ← MIT License
│
├── api/
│   ├── main.py                       ← FastAPI app (v2.0, inference + MLOps routes)
│   └── predict.py                    ← Model loading + single-image inference pipeline
│
├── data/
│   ├── loaders.py                    ← CIFAR-10 DataLoader factory
│   ├── transforms.py                 ← Train augmentations vs test normalization
│   ├── stats.py                      ← Dataset statistics utilities
│   ├── contract.py                   ← Shared constants (split ratio, seed, shape)
│   ├── dataset_stats.json            ← CIFAR-10 channel mean/std values
│   └── __init__.py
│
├── docker/
│   ├── Dockerfile                    ← API container (Python 3.10-slim, multi-module)
│   ├── requirements.txt              ← Docker-specific requirements
│   └── ui/
│       └── Dockerfile                ← Minimal UI container reference
│
├── Evaluation/
│   ├── Evaluation.ipynb              ← Full analysis notebook (sklearn + matplotlib)
│   ├── metrics.py                    ← Evaluation metric utilities
│   ├── plots.py                      ← Chart generation functions
│   ├── report.md                     ← Baseline vs GA-optimized comparison narrative
│   └── results.csv                   ← Raw per-experiment results
│
├── experiments/
│   ├── db_models.py                  ← SQLAlchemy ORM models (4 tables)
│   ├── logger.py                     ← DBLogger (experiment lifecycle)
│   ├── checkpoint.py                 ← DBCheckpointManager (Spot resilience)
│   ├── manager.py                    ← ExperimentManager (run + resume via API)
│   └── __init__.py
│
├── GA module/
│   ├── ga_optimizer_final.py         ← Full DEAP-based GA optimizer (standalone CLI)
│   ├── ga_log.csv                    ← Per-generation convergence log (10 generations)
│   └── README.md                     ← GA module documentation
│
├── model/
│   ├── cnn.py                        ← SimpleCNN architecture
│   ├── train.py                      ← Training loop (AdamW + CosineAnnealing + clipping)
│   ├── evaluate.py                   ← Test-set evaluation utilities
│   ├── run_baseline.py               ← Baseline training script
│   ├── __init__.py
│   └── checkpoints/                  ← Saved .pth model files (gitignored)
│
├── tests/
│   └── test_loaders.py               ← CIFAR-10 DataLoader shape smoke test
│
├── ui/
│   └── dashboard/
│       ├── app.py                    ← Streamlit 4-page dashboard entry point
│       ├── utils.py                  ← API client + CSV loaders
│       ├── charts.py                 ← Plotly chart functions
│       ├── Dockerfile                ← UI container (production build)
│       ├── .dockerignore
│       ├── requirements.txt          ← UI-specific dependencies
│       ├── ga_log.csv                ← GA log copy for dashboard display
│       └── evaluation/
│           └── results.csv           ← Results copy for dashboard display
│
└── docs/
    ├── diagrams/
    │   └── experiment_diagram.html   ← Interactive experiment flow diagram
    ├── reports/
    │   └── cost_analysis.html        ← AWS cloud cost analysis report
    └── proposal/
        └── Project_Idea_Proposal.pdf ← Original project proposal
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `API_URL` | No | API base URL for the dashboard | `http://localhost:8000` |

**`DATABASE_URL` format for AWS RDS:**
```
postgresql://postgres:YOUR_PASSWORD@your-rds-host.rds.amazonaws.com:5432/postgres
```

> **Never commit `.env`** — it is gitignored. Only `.env.example` is tracked in version control.

---

## Installation

### Prerequisites

- Python 3.10+
- Docker + Docker Compose (for containerized deployment)
- AWS account (for EC2 + RDS deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/nour-hatem/Evolutionary-Vision-Optimization-Platform.git
cd Evolutionary-Vision-Optimization-Platform

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install GA dependencies (required for ga_optimizer_final.py)
pip install deap numpy

# 5. Set up environment variables
cp .env.example .env
# Edit .env — set DATABASE_URL (or leave blank to run without DB tracking)
```

---

## Running Locally

### 1. Train the Baseline Model

```bash
python -m model.run_baseline
# Trains for 20 epochs with fixed hyperparameters
# Saves model/checkpoints/baseline_model.pth
# Writes metrics.json to the project root
```

### 2. Run the GA Optimizer

```bash
# Full run (~5–6 hours on CPU; faster on GPU)
python "GA module/ga_optimizer_final.py"

# Resume after a crash or interruption
python "GA module/ga_optimizer_final.py" --resume

# Quick sanity check: 1 generation × 3 individuals (~2 min)
python "GA module/ga_optimizer_final.py" --smoke-test
```

After completion:
- `best_config.json` — winning hyperparameter configuration
- `GA module/ga_log.csv` — per-generation fitness statistics
- `GA module/ga_checkpoint.json` — full population state (gitignored)

### 3. Train the Final Optimized Model

Use the best config from `best_config.json` with the full 15 epochs. The training call with `fast_mode=False` and the config values will produce `model/checkpoints/optimized_model.pth`.

### 4. Start the API

```bash
uvicorn api.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Start the Dashboard

```bash
cd ui/dashboard
streamlit run app.py
# Dashboard: http://localhost:8501
```

### 6. Verify Database Connection

```bash
python check_db.py
# Expected: Connection Successful — lists experiments, run_logs, checkpoints, prediction_logs
```

### 7. Run Tests

```bash
python -m pytest tests/
```

---

## Running with Docker

### Full Stack (API + UI)

```bash
# Copy and fill in environment variables
cp .env.example .env

# Build and start all services
docker-compose up --build -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f

# Check container status
docker-compose ps
```

Services started:
- `cifar-api` at `http://localhost:8000`
- `cifar-ui` at `http://localhost:8501`

### API Container Only

```bash
docker build -f docker/Dockerfile -t cifar-api .
docker run -d -p 8000:8000 --env-file .env cifar-api
```

### UI Container Only

```bash
cd ui/dashboard
docker build -t cifar-ui .
docker run -d -p 8501:8501 -e API_URL=http://localhost:8000 cifar-ui
```

### Container Management

```bash
docker-compose ps             # Status of all services
docker-compose restart api    # Restart API only
docker-compose logs api -f    # Stream API logs
docker-compose down           # Stop and remove containers
docker-compose down -v        # Stop and remove containers + volumes
```

### Healthchecks

Both containers expose Docker healthchecks:

| Container | Probe | Interval |
|-----------|-------|----------|
| `cifar-api` | `curl -f http://localhost:8000/health` | 30s, timeout 5s, 3 retries |
| `cifar-ui` | `curl -f http://localhost:8501/_stcore/health` | 30s, timeout 5s, 3 retries |

---

## Streamlit Dashboard

The dashboard provides four navigation pages:

| Page | Description |
|------|-------------|
| **Home** | Platform overview, key metrics, optimization workflow steps, tech stack |
| **Prediction** | Upload an image, run inference against the deployed API, view predicted class + confidence |
| **Dashboard Analytics** | GA convergence chart, accuracy comparison, confidence distribution, metrics radar |
| **Experiment Logs** | Filterable tables of evaluation results and GA evolution data, CSV export |

### Deploying on Streamlit Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, branch `main`, main file `ui/dashboard/app.py`
4. Add secrets under **App Settings → Secrets**:
   ```toml
   [api]
   url = "http://13.48.139.52:8000"
   ```

---

## AWS Deployment

See [`deployment_guide.md`](deployment_guide.md) for the complete step-by-step guide. Quick summary:

```bash
# 1. Build and push images to DockerHub
docker build -f docker/Dockerfile -t <hub>/cifar-api:latest .
docker push <hub>/cifar-api:latest

cd ui/dashboard
docker build -t <hub>/cifar-ui:latest .
docker push <hub>/cifar-ui:latest

# 2. SSH into EC2
ssh -i <key>.pem ubuntu@<EC2_IP>

# 3. Pull and run containers
docker pull <hub>/cifar-api:latest
docker run -d --name cifar-api --restart unless-stopped -p 8000:8000 \
  -e DATABASE_URL="$DATABASE_URL" <hub>/cifar-api:latest

docker run -d --name cifar-ui --restart unless-stopped -p 8501:8501 \
  -e API_URL="http://<EC2_IP>:8000" <hub>/cifar-ui:latest
```

**Required EC2 Security Group inbound rules:**

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| Custom TCP | TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | TCP | 8501 | 0.0.0.0/0 |
| SSH | TCP | 22 | Your IP only |

**RDS Security Group:** Allow TCP 5432 from EC2's security group.

---

## Experiment Tracking

### Database Schema

```sql
-- Experiment metadata
experiments (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR UNIQUE,
    status VARCHAR,       -- 'running' | 'interrupted' | 'completed'
    config JSONB,
    started_at TIMESTAMP,
    ended_at TIMESTAMP
)

-- Per-generation metrics
run_logs (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER → experiments.id,
    generation INTEGER,
    accuracy FLOAT,
    f1 FLOAT,
    logged_at TIMESTAMP
)

-- GA population checkpoints (enables Spot resume)
checkpoints (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER → experiments.id,
    generation INTEGER,
    population_state JSONB,
    saved_at TIMESTAMP
)

-- Every prediction request
prediction_logs (
    id SERIAL PRIMARY KEY,
    filename VARCHAR,
    prediction VARCHAR,
    confidence FLOAT,
    predicted_at TIMESTAMP
)
```

### Start and Resume Experiments via API

```bash
# Start a new experiment (simulates Spot interruption after 3 generations)
curl -X POST http://localhost:8000/start_experiment \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "spot_demo",
    "max_generations": 6,
    "population_size": 8,
    "mutation_rate": 0.3,
    "dataset": "cifar10",
    "interrupt_after": 3
  }'

# Check status
curl http://localhost:8000/experiment_status/spot_demo

# Resume from last checkpoint
curl -X POST http://localhost:8000/resume_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "spot_demo"}'

# Get per-generation logs
curl http://localhost:8000/get_logs/1
```

---

## Evaluation

Full evaluation results and methodology are in [`Evaluation/`](Evaluation/):

- **`Evaluation/Evaluation.ipynb`** — Jupyter notebook: loads checkpoints, computes sklearn metrics, generates all plots
- **`Evaluation/report.md`** — Narrative comparison of baseline vs GA-optimized results
- **`Evaluation/results.csv`** — Raw per-experiment accuracy, F1 macro, F1 weighted, parameter count, inference time
- **`Evaluation/metrics.py`** — Metric utility functions (accuracy, F1, confusion matrix)
- **`Evaluation/plots.py`** — Chart generation functions (accuracy bar, convergence line, confusion matrices, per-class accuracy)

To regenerate evaluation plots, run `Evaluation/Evaluation.ipynb` after training both models.

### Per-Class Insights (from `Evaluation/report.md`)

- **Best improved class:** `bird` (+8.90% accuracy with GA-optimized model)
- **Lowest accuracy classes (both models):** visually similar pairs — `cat/dog`, `automobile/truck`
- GA convergence shows fast improvement in generations 0–4, then stabilization — a healthy evolutionary pattern

---

## Cost Analysis

Platform runs on **AWS EU-North-1 (Stockholm)**:

| Component | Service | Monthly Cost |
|-----------|---------|-------------|
| Inference API + UI | EC2 `t3.micro` (On-Demand) | ~$7.59 |
| Experiment Database | RDS `db.t3.micro` | ~$15.68 |
| Data Transfer | Minimal outbound | ~$1.00 |
| **Total** | | **~$24.27/month** |

### Cost Optimization with Spot Instances

| Strategy | Use Case | Savings |
|----------|----------|---------|
| EC2 On-Demand | Always-on API + UI serving | Baseline |
| EC2 Spot | Long GA training runs | ~70% cheaper per hour |
| AWS Fargate Spot | Short-to-medium experiments | Pay only during job execution |
| **Hybrid (Recommended)** | On-Demand API + Fargate Spot for training | **~30% total savings** |

The checkpoint system is the **key enabler** of Spot cost savings — every GA generation is saved to RDS so a Spot interruption loses at most one generation of compute.

> See [`docs/reports/cost_analysis.html`](docs/reports/cost_analysis.html) for the full interactive cost report.

---

## Screenshots

### Evaluation Plots (Generated — Available in `Evaluation/Plots/`)

The following evaluation charts are pre-generated and available in the repository:

| Plot | File | Description |
|------|------|-------------|
| Accuracy & F1 Comparison | `Evaluation/Plots/01_accuracy_f1.png` | Baseline vs GA-Optimized bar chart |
| GA Convergence | `Evaluation/Plots/02_ga_convergence.png` | Best and average fitness per generation |
| Confusion Matrix — Baseline | `Evaluation/Plots/03_confusion_matrix_baseline.png` | Per-class predictions for baseline model |
| Confusion Matrix — Optimized | `Evaluation/Plots/04_confusion_matrix_optimized.png` | Per-class predictions for GA-optimized model |
| Per-Class Accuracy | `Evaluation/Plots/05_per_class_accuracy.png` | Accuracy for all 10 CIFAR-10 classes |

### Deployment Screenshots 

> *Capture these from the live AWS deployment and add to `screenshots/` folder.*

| Screenshot | File | What to Capture |
|---|---|---|
| Dashboard Home | `screenshots/01_dashboard_home.png` | Streamlit Home page with metrics tiles |
| Prediction Result | `screenshots/02_prediction_result.png` | Prediction page with uploaded image + result |
| GA Convergence Chart | `screenshots/03_ga_convergence.png` | GA convergence chart from Dashboard Analytics |
| Model Comparison | `screenshots/04_model_comparison.png` | Accuracy comparison bar chart |
| Experiment Logs | `screenshots/05_experiment_logs.png` | Experiment Logs table |
| API Swagger Docs | `screenshots/06_api_docs.png` | FastAPI Swagger UI at `/docs` |
| Health Endpoint | `screenshots/07_health_endpoint.png` | `/health` JSON response |
| EC2 Running | `screenshots/08_ec2_running.png` | AWS EC2 console — running instance |
| RDS Connected | `screenshots/09_rds_connected.png` | `check_db.py` output showing 4 DB tables |
| Docker Running | `screenshots/10_docker_ps.png` | `docker ps` — both containers healthy |

---

## Cloud Computing Bonus

This project implements an advanced cloud-computing bonus focused on scalable and reliable AI experiment orchestration. Each evolutionary optimization experiment is treated as a **managed workload** with structured execution, per-generation checkpointing, persistent database logging, and real-time analytics monitoring.

### Bonus Features Implemented

#### 1. Managed Experiment Runner
The `ExperimentManager` class (`experiments/manager.py`) provides structured experiment execution via REST API endpoints (`POST /start_experiment`, `POST /resume_experiment`). Multiple optimization experiments can be launched, monitored, and managed without touching the server directly.

#### 2. Experiment Checkpointing and Recovery (Spot-Resilient)
The `DBCheckpointManager` (`experiments/checkpoint.py`) saves the full GA population state as JSONB to AWS RDS PostgreSQL after **every generation**. If a Spot instance is reclaimed mid-run, `POST /resume_experiment` loads the last checkpoint and continues from `last_generation + 1` — no computation is lost.

```bash
# Demonstrate Spot interruption and resume
curl -X POST http://localhost:8000/start_experiment \
  -d '{"run_name": "spot_demo", "max_generations": 6, "interrupt_after": 3}'

# Resume after interruption
curl -X POST http://localhost:8000/resume_experiment \
  -d '{"run_name": "spot_demo"}'
```

The standalone GA also supports resumption:
```bash
python "GA module/ga_optimizer_final.py" --resume
```

#### 3. Batch/Spot-Style Execution Analysis
Cost analysis (`docs/reports/cost_analysis.html`) quantifies cloud execution strategies:

| Strategy | Use Case | Savings vs On-Demand |
|----------|----------|----------------------|
| EC2 On-Demand | Always-on API + UI | Baseline |
| EC2 Spot | Long GA training runs | ~70% cheaper per hour |
| AWS Fargate Spot | Short-medium experiments | Pay only during job execution |
| **Hybrid (Recommended)** | On-Demand API + Fargate Spot training | **~30% total savings** |

#### 4. PostgreSQL Cloud Database Logging (AWS RDS)

All experiment data is persisted to **AWS RDS PostgreSQL** across four normalized tables:

```sql
experiments      -- run metadata, status (running/interrupted/completed), config (JSONB), timestamps
run_logs         -- per-generation accuracy and F1 metrics
checkpoints      -- full GA population snapshots enabling Spot-resume (JSONB)
prediction_logs  -- every inference request: filename, class, confidence, timestamp
```

Verify connectivity:
```bash
python check_db.py
# Expected: Connection Successful — lists all 4 tables
```

#### 5. Advanced Dashboard Analytics

The 4-page Streamlit dashboard (`ui/dashboard/app.py`) provides full experiment observability:

| Page | Analytics Provided |
|------|--------------------|
| **Home** | Platform KPIs, optimization workflow, tech stack |
| **Prediction** | Live image inference with class and confidence score |
| **Dashboard Analytics** | GA convergence chart, accuracy comparison bar chart, metrics radar (Plotly) |
| **Experiment Logs** | Per-generation filterable tables, evaluation results, CSV export |

### Cloud Computing Contribution Summary

| Capability | Implementation |
|-----------|---------------|
| Scalable AI workload orchestration | REST API + ExperimentManager |
| Cloud-ready experiment management | AWS RDS PostgreSQL persistence |
| Operational reliability | Docker healthchecks + `restart: unless-stopped` |
| Resumable workloads | Per-generation checkpoint to RDS + `--resume` flag |
| Persistent experiment tracking | 4-table PostgreSQL schema via SQLAlchemy ORM |
| Analytics-driven monitoring | Streamlit + Plotly interactive dashboard |
| Cost-aware execution | Spot/Fargate analysis, ~70% training cost savings |
| Centralized logging | All events logged: experiments, generations, predictions |

---

## Future Improvements

| Priority | Improvement | Rationale |
|----------|-------------|-----------|
| High | **Real GA integration** — replace mock `ExperimentManager` loop with live `run_ga()` calls | Connect the API's experiment tracking to the actual DEAP optimizer |
| High | **AWS Batch + Fargate** — submit GA runs as managed batch jobs | Auto-scaling, pay-per-job, no idle EC2 cost during training |
| High | **S3 model registry** — store `.pth` checkpoints in S3 with presigned URL download | Remove large binaries from filesystem, enable model versioning |
| Medium | **JWT authentication** — protect `/start_experiment` and `/resume_experiment` | Prevent unauthorized experiment launches in public deployment |
| Medium | **Restrict CORS** — change `allow_origins=["*"]` to specific UI origin | Harden API security for production |
| Medium | **Multi-dataset support** — extend GA search to CIFAR-100 | Generalize the platform beyond CIFAR-10 |
| Low | **Transfer learning** — add ResNet/EfficientNet backbone option to SimpleCNN | Potentially push accuracy above 95% |
| Low | **Bayesian search comparison** — compare GA vs Optuna/Hyperopt on same search space | Validate GA as a competitive strategy |



## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

Copyright (c) 2026 Nour Hatem

---

<div align="center">

**Cloud Computing · Evolutionary Algorithms · Computer Vision · MLOps**

*Spring 2026 — Evolutionary Vision Optimization Platform*

</div>
