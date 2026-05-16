<div align="center">

# 🧠 Evolutionary Vision Optimization Platform

**A cloud-native research platform that combines Genetic Algorithm-driven hyperparameter optimization with Convolutional Neural Networks to achieve state-of-the-art image classification on CIFAR-10 — deployed on AWS with full MLOps experiment tracking.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20RDS-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-RDS-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://aws.amazon.com/rds/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Cloud Infrastructure](#-cloud-infrastructure)
- [Genetic Algorithm + CNN Workflow](#-genetic-algorithm--cnn-workflow)
- [API Endpoints](#-api-endpoints)
- [Folder Structure](#-folder-structure)
- [Environment Variables](#-environment-variables)
- [Installation](#-installation)
- [Running Locally](#-running-locally)
- [Running with Docker](#-running-with-docker)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [AWS Deployment](#-aws-deployment)
- [Experiment Tracking](#-experiment-tracking)
- [Cost Analysis](#-cost-analysis)
- [Example Outputs](#-example-outputs)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)

---

## 🔭 Overview

The **Evolutionary Vision Optimization Platform** is an end-to-end cloud computing research project that solves two intertwined problems:

1. **How to automatically find the best CNN architecture** for CIFAR-10 image classification — without manual trial-and-error — using a Genetic Algorithm.
2. **How to run long, expensive training jobs on the cloud cheaply** — using AWS Spot instances with automatic checkpoint/resume so that an interrupted job loses zero progress.

The platform consists of three tightly integrated components:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Evolutionary Optimizer** | DEAP + PyTorch | Genetic Algorithm that evolves CNN hyperparameters |
| **Inference + MLOps API** | FastAPI + SQLAlchemy | REST API for predictions and experiment tracking |
| **Monitoring Dashboard** | Streamlit + Plotly | Real-time visualization of experiments and predictions |

All three run as Docker containers on AWS EC2, with experiment data persisted to **AWS RDS PostgreSQL** — including full checkpoint state so Spot instance interruptions are fully recoverable.

> 📄 **Project Proposal:** [`docs/proposal/Project_Idea_Proposal.pdf`](docs/proposal/Project_Idea_Proposal.pdf)

---

## 📊 Results

The GA-optimized model significantly outperforms the hand-tuned baseline:

| Metric | Baseline CNN | GA-Optimized CNN | Improvement |
|--------|-------------|-----------------|-------------|
| **Accuracy** | 85.87% | **89.93%** | **+4.06%** |
| **F1 Macro** | 85.79% | **89.87%** | **+4.08%** |
| **F1 Weighted** | 85.79% | 89.87% | +4.08% |
| Inference (ms/batch) | 56.73 ms | 48.84 ms | −7.89 ms |
| Parameters | 3.4M | 12.2M | +8.9M |

### GA Convergence

The GA ran for **10 generations** with a population of **20 individuals**:
- **Gen 0 best fitness:** 70.02%
- **Gen 10 best fitness:** 73.38% (during 5-epoch fast search)
- **Average fitness** across population rose from 56.97% → 69.54% (whole population improved, not just the winner)

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

---

## ✨ Features

### Machine Learning
- 🧬 **Genetic Algorithm** hyperparameter search over 5 dimensions (lr, filters, layers, dropout, batch size)
- 🧠 **SimpleCNN** — configurable architecture with conv blocks, batch norm, dropout, and adaptive FC head
- ⚡ **Fast-mode training** (3–5 epochs) during GA search for speed; full 15–20 epoch training for final model
- 📈 **Cosine annealing** learning rate scheduler + gradient clipping for stable convergence
- 📐 **Kaiming/Xavier weight initialization** for faster convergence

### MLOps & Cloud
- 🔄 **Spot-resilient checkpointing** — GA population state saved to AWS RDS after every generation; resumes seamlessly after interruption
- 📊 **Full experiment tracking** — every run, generation, and prediction logged to PostgreSQL
- 🔁 **Start/resume experiments** via REST API without touching the server
- 🐳 **Docker Compose** orchestration — single command to run the entire platform

### API & Dashboard
- 🖼️ **Image classification API** — upload any image, get CIFAR-10 class + confidence score
- 🛡️ **MIME type validation** — rejects non-image uploads with HTTP 415
- 📋 **Experiment status API** — query any run's status, logs, and checkpoint in real time
- 📊 **Interactive Streamlit dashboard** — GA convergence charts, accuracy comparisons, prediction history
- 🌐 **CORS-enabled** — UI container can call API across Docker network

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AWS EC2 (t3.micro)                    │
│                                                          │
│  ┌─────────────────┐       ┌─────────────────────────┐  │
│  │  Streamlit UI   │──────▶│     FastAPI Backend     │  │
│  │   :8501         │       │        :8000            │  │
│  │                 │       │                         │  │
│  │  • Home         │       │  POST /predict          │  │
│  │  • Prediction   │       │  POST /start_experiment │  │
│  │  • Analytics    │       │  POST /resume_experiment│  │
│  │  • Logs         │       │  GET  /get_logs/{id}    │  │
│  └─────────────────┘       │  GET  /experiment_status│  │
│                             └──────────┬────────────┘  │
│                                        │                │
└────────────────────────────────────────│────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   AWS RDS            │
                              │   PostgreSQL         │
                              │   (db.t3.micro)      │
                              │                      │
                              │  • experiments       │
                              │  • run_logs          │
                              │  • checkpoints       │
                              │  • prediction_logs   │
                              └─────────────────────┘
```

> 🔬 **Interactive Experiment Flow Diagram:** [`docs/diagrams/experiment_diagram.html`](docs/diagrams/experiment_diagram.html)

---

## 🛠️ Tech Stack

### Backend
| Library | Version | Role |
|---------|---------|------|
| **FastAPI** | ≥ 0.111 | REST API framework |
| **Uvicorn** | ≥ 0.29 | ASGI server |
| **PyTorch** | ≥ 2.2 | CNN training and inference |
| **Torchvision** | ≥ 0.17 | CIFAR-10 datasets and transforms |
| **Pillow** | ≥ 10.3 | Image decoding for inference |
| **SQLAlchemy** | ≥ 2.0 | ORM for PostgreSQL |
| **psycopg2-binary** | ≥ 2.9 | PostgreSQL adapter |

### Evolutionary Optimizer
| Library | Version | Role |
|---------|---------|------|
| **DEAP** | any | Genetic Algorithm framework |
| **NumPy** | ≥ 1.24 | Fitness statistics |

### Frontend
| Library | Version | Role |
|---------|---------|------|
| **Streamlit** | ≥ 1.32 | Multi-page dashboard |
| **Plotly** | ≥ 5.18 | Interactive charts |
| **Pandas** | ≥ 2.0 | Data manipulation |
| **Requests** | ≥ 2.31 | API communication |

### Infrastructure
| Service | Purpose |
|---------|---------|
| **AWS EC2 t3.micro** | API + UI containers |
| **AWS RDS db.t3.micro** | PostgreSQL experiment database |
| **Docker + Compose** | Container orchestration |
| **DockerHub** | Container image registry |

---

## ☁️ Cloud Infrastructure

Deployed on **AWS EU-North-1 (Stockholm)**:

```
                    Internet
                        │
              ┌─────────▼──────────┐
              │   EC2 Security     │
              │   Group (ports     │
              │   8000, 8501)      │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │   EC2 t3.micro     │
              │   Ubuntu 22.04     │
              │                    │
              │  ┌──────────────┐  │
              │  │ cifar-api    │  │──────▶ RDS PostgreSQL
              │  │ :8000        │  │        db.t3.micro
              │  └──────────────┘  │
              │  ┌──────────────┐  │
              │  │ cifar-ui     │  │
              │  │ :8501        │  │
              │  └──────────────┘  │
              └────────────────────┘
```

Both containers run with `--restart unless-stopped` for automatic recovery after instance reboots.

---

## 🧬 Genetic Algorithm + CNN Workflow

The platform implements a complete evolutionary hyperparameter optimization pipeline:

### Step 1 — Chromosome Encoding

Each candidate configuration is encoded as a **5-gene chromosome**:

```
Chromosome = [lr_idx, filter_idx, layer_idx, dropout_idx, batch_idx]

Gene ranges:
  lr_idx     → [0.01, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001]  (7 values)
  filter_idx → [32, 64, 96, 128]                                      (4 values)
  layer_idx  → [2, 3, 4, 5]                                           (4 values)
  dropout_idx→ [0.3, 0.4, 0.5]                                        (3 values)
  batch_idx  → [32, 64, 128]                                           (3 values)
```

### Step 2 — Fitness Evaluation

Each individual is evaluated by calling `model/train.py` with `fast_mode=True` (5 epochs) and returning the validation accuracy as fitness.

### Step 3 — Evolution Loop

```
Generation 0: Evaluate all 20 random individuals
For each generation 1..N:
  1. Elitism:   carry best 1 individual unchanged
  2. Selection: tournament selection (size 3) for remaining 19
  3. Crossover: two-point crossover (70% probability per pair)
  4. Mutation:  random gene replacement (30% per individual, 30% per gene)
  5. Evaluate:  only re-evaluate changed individuals
  6. Checkpoint: save full population state to AWS RDS
```

### Step 4 — Spot-Resilient Checkpointing

Every generation is saved to `RDS.checkpoints`. If the Spot instance is reclaimed:

```
Experiment interrupted at Gen N
        ↓
New instance starts
        ↓
POST /resume_experiment {"run_name": "my_run"}
        ↓
ExperimentManager.resume():
  load_latest_checkpoint(exp_id) → returns (N, population_state)
  continue training from Gen N+1
        ↓
Experiment completes as if never interrupted
```

### Step 5 — Final Training

The winning chromosome is decoded and re-trained for **15 full epochs** to produce the final `optimized_model.pth`.

---

## 🔌 API Endpoints

Base URL: `http://<HOST>:8000`  
Interactive docs: `http://<HOST>:8000/docs`

### Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info and version |
| `GET` | `/health` | Liveness probe with DB connection status |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload an image file → returns predicted CIFAR-10 class + confidence |

**Request** (multipart/form-data):
```
file: <image file>   # JPEG, PNG, BMP, WEBP, etc.
```

**Response:**
```json
{
  "prediction": "airplane",
  "confidence": 0.9342
}
```

### MLOps — Experiment Tracking

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/start_experiment` | Start a new GA optimization run (async background task) |
| `POST` | `/resume_experiment` | Resume an interrupted experiment from its last RDS checkpoint |
| `GET` | `/get_logs/{experiment_id}` | Fetch per-generation metrics for a running or completed experiment |
| `GET` | `/experiment_status/{run_name}` | Full status including config, timings, logs, and checkpoint info |

**Start experiment request:**
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

---

## 📁 Folder Structure

```
Evolutionary-Vision-Optimization-Platform/
│
├── 📄 README.md                         ← This file
├── 📄 requirements.txt                  ← Python dependencies
├── 📄 docker-compose.yml                ← Multi-service orchestration
├── 📄 .env.example                      ← Environment variable template
├── 📄 best_config.json                  ← GA-discovered best hyperparameters
├── 📄 metrics.json                      ← Baseline training metrics
├── 📄 check_db.py                       ← AWS RDS connection verifier
│
├── 🔌 api/
│   ├── main.py                          ← FastAPI v2.0 app (inference + MLOps routes)
│   ├── predict.py                       ← Model loading + single-image inference
│   └── __init__.py
│
├── 🗃️ data/
│   ├── loaders.py                       ← CIFAR-10 DataLoader factory
│   ├── transforms.py                    ← Train/test augmentation pipelines
│   ├── stats.py                         ← Dataset statistics utilities
│   ├── contract.py                      ← Shared constants (split ratio, seed)
│   ├── dataset_stats.json               ← CIFAR-10 normalization values
│   └── __init__.py
│
├── 🐳 docker/
│   ├── Dockerfile                       ← API container (multi-module build)
│   ├── requirements.txt                 ← Docker-specific dependencies
│   └── ui/
│       └── Dockerfile                   ← Minimal UI container reference
│
├── 📊 Evaluation/
│   ├── Evaluation.ipynb                 ← Full analysis notebook
│   ├── metrics.py                       ← Evaluation metric utilities
│   ├── plots.py                         ← Chart generation functions
│   ├── report.md                        ← Baseline vs GA-optimized summary
│   ├── results.csv                      ← Raw experiment results
│   └── Plots/                           ← Pre-generated evaluation charts
│       ├── 01_accuracy_f1.png
│       ├── 02_ga_convergence.png
│       ├── 03_confusion_matrix_baseline.png
│       ├── 04_confusion_matrix_optimized.png
│       └── 05_per_class_accuracy.png
│
├── 🧪 experiments/
│   ├── db_models.py                     ← SQLAlchemy ORM models
│   ├── logger.py                        ← DBLogger (experiment lifecycle)
│   ├── checkpoint.py                    ← DBCheckpointManager (Spot resilience)
│   ├── manager.py                       ← ExperimentManager (run + resume)
│   └── __init__.py
│
├── 🧬 GA module/
│   ├── ga_optimizer_final.py            ← Full DEAP-based GA optimizer
│   ├── ga_log.csv                       ← Per-generation convergence log
│   ├── ga_checkpoint.json               ← Last population state
│   └── README.md                        ← GA module documentation
│
├── 🧠 model/
│   ├── cnn.py                           ← SimpleCNN architecture
│   ├── train.py                         ← Training loop (AdamW + CosineAnnealing)
│   ├── evaluate.py                      ← Test-set evaluation utilities
│   ├── run_baseline.py                  ← Baseline training script
│   ├── __init__.py
│   └── checkpoints/                     ← Saved .pth files (gitignored)
│
├── 🧪 tests/
│   └── test_loaders.py                  ← CIFAR-10 DataLoader shape test
│
├── 🖥️ ui/
│   └── dashboard/
│       ├── app.py                       ← Streamlit multi-page dashboard
│       ├── utils.py                     ← API communication + CSV loading
│       ├── charts.py                    ← Plotly chart functions
│       ├── Dockerfile                   ← UI container (production)
│       ├── .dockerignore
│       ├── requirements.txt
│       ├── ga_log.csv                   ← GA log for dashboard display
│       └── evaluation/
│           └── results.csv              ← Results for dashboard display
│
└── 📚 docs/
    ├── diagrams/
    │   └── experiment_diagram.html      ← Interactive experiment flow diagram
    ├── reports/
    │   └── cost_analysis.html           ← AWS cloud cost analysis report
    └── proposal/
        └── Project_Idea_Proposal.pdf    ← Original project proposal document
```

---

## 🔐 Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `API_URL` | No | API base URL for the dashboard | `http://localhost:8000` |

> ⚠️ **Never commit `.env`** — it is gitignored. Use `.env.example` as the safe template.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Docker + Docker Compose (for containerized deployment)
- AWS account (for RDS + EC2 deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/nour-hatem/Evolutionary-Vision-Optimization-Platform.git
cd Evolutionary-Vision-Optimization-Platform

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env — set DATABASE_URL (or leave blank to run without DB tracking)
```

---

## 🚀 Running Locally

### 1. Train the Baseline Model

```bash
python -m model.run_baseline
# Trains for 20 epochs, saves model/checkpoints/baseline_model.pth
# Also writes metrics.json to the project root
```

### 2. Run the GA Optimizer

```bash
# Full run (~5–6 hours on CPU, faster on GPU)
python "GA module/ga_optimizer_final.py"

# Resume after a crash
python "GA module/ga_optimizer_final.py" --resume

# Quick sanity check (1 gen × 3 individuals)
python "GA module/ga_optimizer_final.py" --smoke-test
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Start the Dashboard

```bash
cd ui/dashboard
streamlit run app.py
# Dashboard available at http://localhost:8501
```

### 5. Verify Database Connection

```bash
python check_db.py
```

---

## 🐳 Running with Docker

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
```

### API Only

```bash
docker build -f docker/Dockerfile -t cifar-api .
docker run -d -p 8000:8000 --env-file .env cifar-api
```

### UI Only

```bash
cd ui/dashboard
docker build -t cifar-ui .
docker run -d -p 8501:8501 -e API_URL=http://localhost:8000 cifar-ui
```

### Container Management

```bash
docker-compose ps          # Check status
docker-compose restart api # Restart API
docker-compose down        # Stop everything
```

---

## 📊 Streamlit Dashboard

The dashboard provides four navigation pages:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Platform overview, key metrics, workflow steps, tech stack |
| 🔍 **Prediction** | Upload an image → run inference → view predicted class + confidence |
| 📊 **Dashboard Analytics** | GA convergence chart, accuracy comparison, confidence distributions |
| 📋 **Experiment Logs** | Filterable tables of evaluation results and GA evolution data, CSV export |

To deploy the dashboard on **Streamlit Cloud**:

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, branch `main`, main file `ui/dashboard/app.py`
4. Add secrets:
   ```toml
   [api]
   url = "https://your-api-endpoint.com"
   ```

---

## ☁️ AWS Deployment

See [`deployment_guide.md`](deployment_guide.md) for the complete step-by-step guide. Quick summary:

```bash
# 1. Build and push images to DockerHub
docker build -f docker/Dockerfile -t <hub>/cifar-api:latest .
docker push <hub>/cifar-api:latest

# 2. SSH into EC2
ssh -i <key>.pem ubuntu@<EC2_IP>

# 3. Pull and run containers
docker pull <hub>/cifar-api:latest
docker run -d --name cifar-api --restart unless-stopped -p 8000:8000 \
  -e DATABASE_URL="$DATABASE_URL" <hub>/cifar-api:latest

docker run -d --name cifar-ui --restart unless-stopped -p 8501:8501 \
  <hub>/cifar-ui:latest
```

Required EC2 Security Group inbound rules: TCP 8000, TCP 8501.

---

## 🔬 Experiment Tracking

The platform tracks every experiment and prediction in AWS RDS PostgreSQL:

### Database Schema

```sql
experiments     -- experiment metadata (name, config, status, timing)
run_logs        -- per-generation metrics (accuracy, F1 per generation)
checkpoints     -- GA population state (enables Spot resume)
prediction_logs -- every /predict call (filename, class, confidence)
```

### Start and Resume Experiments

```bash
# Start a new experiment (will simulate Spot interruption after 3 generations)
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

# Resume from checkpoint
curl -X POST http://localhost:8000/resume_experiment \
  -H "Content-Type: application/json" \
  -d '{"run_name": "spot_demo"}'
```

---

## 💰 Cost Analysis

The platform runs on **AWS EU-North-1 (Stockholm)**:

| Component | Service | Monthly Cost |
|-----------|---------|-------------|
| Inference API + UI | EC2 `t3.micro` (On-Demand) | ~$7.59 |
| Experiment Database | RDS `db.t3.micro` | ~$15.68 |
| Data Transfer | Minimal | ~$1.00 |
| **Total** | | **~$24.27/month** |

### Cost Optimization with Spot Instances

| Strategy | Use Case | Savings |
|----------|----------|---------|
| EC2 On-Demand | Always-on API + UI | Baseline |
| EC2 Spot | Long GA training runs | ~70% cheaper per hour |
| AWS Fargate Spot | Short-to-medium experiments | Pay only during job execution |
| **Hybrid (Recommended)** | On-Demand API + Fargate Spot for training | **~30% total savings** |

The checkpoint system is the **key enabler** for Spot cost optimization — every generation is saved to RDS, so a Spot interruption loses at most one generation of work.

> 📊 **Full interactive cost report:** [`docs/reports/cost_analysis.html`](docs/reports/cost_analysis.html)

---

## 🖼️ Example Outputs

### Image Prediction

```bash
curl -X POST http://localhost:8000/predict -F "file=@airplane.jpg"
```

```json
{
  "prediction": "airplane",
  "confidence": 0.9342
}
```

### Experiment Status

```bash
curl http://localhost:8000/experiment_status/run_v1
```

```json
{
  "experiment_id": 1,
  "run_name": "run_v1",
  "status": "completed",
  "generations_completed": 10,
  "logs": [
    {"gen": 0,  "accuracy": 0.7002, "f1": 0.6980},
    {"gen": 5,  "accuracy": 0.7221, "f1": 0.7199},
    {"gen": 10, "accuracy": 0.7338, "f1": 0.7319}
  ],
  "checkpoint": null
}
```

### Evaluation Charts

<div align="center">

| Accuracy & F1 Comparison | GA Convergence |
|:---:|:---:|
| ![Accuracy](Evaluation/Plots/01_accuracy_f1.png) | ![GA](Evaluation/Plots/02_ga_convergence.png) |

| Baseline Confusion Matrix | Optimized Confusion Matrix |
|:---:|:---:|
| ![Baseline CM](Evaluation/Plots/03_confusion_matrix_baseline.png) | ![Optimized CM](Evaluation/Plots/04_confusion_matrix_optimized.png) |

</div>

---

## 🔮 Future Improvements

| Priority | Improvement | Rationale |
|----------|-------------|-----------|
| High | **Real GA integration** — replace mock `ExperimentManager` loop with live `run_ga()` calls | Connect the API's experiment tracking to the actual GA optimizer |
| High | **AWS Batch + Fargate** — submit GA runs as managed batch jobs | Auto-scaling, pay-per-job, no idle EC2 cost |
| High | **S3 model registry** — store `.pth` files in S3 with presigned URL download | Remove large binaries from filesystem, enable model versioning |
| Medium | **JWT authentication** — protect `/start_experiment` and `/resume_experiment` | Prevent unauthorized experiment launches in production |
| Medium | **Restrict CORS** — change `allow_origins=["*"]` to specific UI origin | Harden API security for public deployment |
| Medium | **Multi-dataset support** — extend GA search to MNIST, CIFAR-100 | Generalize the platform beyond CIFAR-10 |
| Low | **Transfer learning** — add ResNet/EfficientNet backbone options to SimpleCNN | Potentially push accuracy above 95% |
| Low | **Hyperparameter Bayesian search** — compare GA vs Optuna/Hyperopt | Validate GA as a competitive optimization strategy |

---

## 👥 Contributors

| Name | Role | Module |
|------|------|--------|
| **Yassin** | CNN Architecture & Training | `model/cnn.py`, `model/train.py`, `model/evaluate.py` |
| **Ahmad** | Genetic Algorithm Optimizer | `GA module/ga_optimizer_final.py` |
| **Nour** | Data Pipeline & Dashboard | `data/`, `ui/dashboard/` |
| **Mahmoud** | API & Cloud Deployment | `api/`, `docker/`, `experiments/`, AWS EC2/RDS |

---

## 📄 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Cloud Computing · Evolutionary Algorithms · Computer Vision · MLOps**

*Spring 2026 — Evolutionary Vision Optimization Platform*

</div>
