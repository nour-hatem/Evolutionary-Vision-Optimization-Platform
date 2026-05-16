# Evolutionary Vision Optimization — Deployment Guide

> Complete step-by-step guide: Docker build → AWS EC2 deployment → testing.
> All sensitive values (IP addresses, passwords, instance IDs) must be stored
> in your `.env` file — **never committed to version control**.

## Project Architecture

```
User (Browser)
   │
   ▼
Streamlit UI  (:8501)
   │
   ▼
FastAPI API   (:8000)
   ├── CNN Model (PyTorch)
   ├── GA Optimizer (DEAP)
   └── AWS RDS PostgreSQL
```

---

## Bugs Fixed During Migration

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `model/train.py` | Duplicate `train()` function shadowing the real one | Removed duplicate |
| 2 | `docker/Dockerfile` | Two `FROM` statements + missing COPY commands | Rewritten with single `FROM` |
| 3 | `data/loaders.py` | `get_loaders(batch_size)` vs `get_loaders(dataset, batch_size)` mismatch | Unified signature |
| 4 | `ui/dashboard/utils.py` | Hardcoded EC2 IP instead of env var | Uses `API_URL` env var |
| 5 | `ui/dashboard/utils.py` | Dead exception handlers after broad `except Exception` | Reordered: specific first |
| 6 | `experiments/db_models.py` | Hardcoded DB credentials in source code | Reads from `DATABASE_URL` env var |
| 7 | `api/main.py` | No CORS middleware — UI couldn't call API cross-origin | Added `CORSMiddleware` |
| 8 | `api/main.py` | DB session leak in `/get_logs` | Added `try/finally db.close()` |

---

## Phase 1: Environment Setup

```bash
cp .env.example .env
# Edit .env and fill in your real DATABASE_URL and EC2 IP
```

---

## Phase 2: Build Docker Images (local machine)

```powershell
# Build API image
docker build -f docker/Dockerfile -t <your-dockerhub>/cifar-api:latest .

# Build UI image
cd ui\dashboard
docker build -t <your-dockerhub>/cifar-ui:latest .
cd ..\..
```

### Test locally (optional but recommended)

```powershell
docker run -d --name test-api -p 8000:8000 --env-file .env <your-dockerhub>/cifar-api:latest
curl http://localhost:8000/health

docker run -d --name test-ui -p 8501:8501 <your-dockerhub>/cifar-ui:latest
# Open http://localhost:8501

docker stop test-api test-ui && docker rm test-api test-ui
```

---

## Phase 3: Push Docker Images to DockerHub

```powershell
docker login -u <your-dockerhub>
docker push <your-dockerhub>/cifar-api:latest
docker push <your-dockerhub>/cifar-ui:latest
```

---

## Phase 4: Verify Database Connection

```powershell
# On local machine with .env loaded
python check_db.py
```

Expected output:
```
Connecting to AWS RDS...

Connection Successful!
------------------------------
Found the following tables:
  experiments
  run_logs
  checkpoints
  prediction_logs
```

---

## Phase 5: Deploy to EC2

### 5.1 — SSH into EC2

```bash
ssh -i <your-pem>.pem ubuntu@<EC2_IP>
```

### 5.2 — Install Docker (if not already)

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER
sudo systemctl enable docker && sudo systemctl start docker
exit  # re-login for group changes
```

### 5.3 — Pull and run containers

```bash
docker login -u <your-dockerhub>
docker pull <your-dockerhub>/cifar-api:latest
docker pull <your-dockerhub>/cifar-ui:latest

# Run API (pass DATABASE_URL from environment)
docker run -d \
  --name cifar-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e DATABASE_URL="$DATABASE_URL" \
  <your-dockerhub>/cifar-api:latest

# Run UI
docker run -d \
  --name cifar-ui \
  --restart unless-stopped \
  -p 8501:8501 \
  <your-dockerhub>/cifar-ui:latest
```

---

## Phase 6: EC2 Security Group

Add inbound rules:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| Custom TCP | TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | TCP | 8501 | 0.0.0.0/0 |

---

## Phase 7: Test Everything

```
http://<EC2_IP>:8000/health   → {"status":"ok","database":"connected"}
http://<EC2_IP>:8000/docs     → Swagger UI
http://<EC2_IP>:8501          → Streamlit dashboard
```

---

## Quick Update Workflow

```powershell
# Rebuild images
docker build -f docker/Dockerfile -t <your-dockerhub>/cifar-api:latest .
docker push <your-dockerhub>/cifar-api:latest
```

```bash
# On EC2: pull and restart
docker pull <your-dockerhub>/cifar-api:latest
docker stop cifar-api && docker rm cifar-api
docker run -d --name cifar-api --restart unless-stopped -p 8000:8000 \
  -e DATABASE_URL="$DATABASE_URL" <your-dockerhub>/cifar-api:latest
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` on port 8000/8501 | Check EC2 security group inbound rules |
| API can't connect to RDS | Check RDS security group allows EC2's security group on port 5432 |
| `No model checkpoint found` | Run `python -m model.run_baseline` to train and save a checkpoint |
| Docker build fails on `psycopg2` | Uses `psycopg2-binary` which avoids libpq; if issue persists add `libpq-dev` to apt |
| UI shows "Cannot reach API" | Verify API is running (`docker ps`) and port 8000 is open |
| `EnvironmentError: DATABASE_URL not set` | Ensure `.env` is present and `DATABASE_URL` is filled in |
