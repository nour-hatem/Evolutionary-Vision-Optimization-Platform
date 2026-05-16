# Security Report

**Analysis Date:** 2026-05-16  
**Scope:** SOURCE (`Cloud_Project_Final`) and TARGET (`Evolutionary-Vision-Optimization-Platform`)  
**Status:** All identified issues resolved

---

## Critical Findings (RESOLVED)

### CRIT-01: AWS EC2 Private Key in Source Repository
| | |
|---|---|
| **File** | `Cloud_Project_Final/Cloud_Project/Cloud26.pem` |
| **Type** | AWS EC2 SSH Private Key (RSA 2048-bit) |
| **Risk** | Anyone with access to this file can SSH into the EC2 instance as `ubuntu` |
| **Action** | **NOT TRANSFERRED** — Private key files must never be in version control |
| **Recommendation** | Rotate the EC2 key pair immediately; generate a new key pair and replace the PEM |

---

### CRIT-02: Database Password in Source Code
| | |
|---|---|
| **File** | `experiments/db_models.py` (SOURCE) |
| **Exposed value** | `postgresql://postgres:V3lvet%23T0rnado@clouddb.c1yu626sut0q.eu-north-1.rds.amazonaws.com:5432/postgres` |
| **Risk** | Full RDS database access; password exposed in source code and git history |
| **Action** | Credential removed. TARGET `experiments/db_models.py` reads exclusively from `DATABASE_URL` environment variable |
| **Recommendation** | Rotate the RDS database password; revoke public accessibility of RDS if not required |

---

### CRIT-03: Database Password in Docker Compose
| | |
|---|---|
| **File** | `docker-compose.yml` (SOURCE) |
| **Exposed value** | `DATABASE_URL=postgresql://postgres:V3lvet%23T0rnado@...` |
| **Risk** | DB credentials committed to any repository using this file |
| **Action** | TARGET `docker-compose.yml` uses `DATABASE_URL=${DATABASE_URL}` with `env_file: .env` |
| **Recommendation** | Never use literal credentials in `docker-compose.yml`; always use env var substitution |

---

### CRIT-04: EC2 IP Address in Application Code
| | |
|---|---|
| **File** | `ui/dashboard/utils.py` (SOURCE) |
| **Exposed value** | `API_URL = "http://13.51.70.11:8000/predict"` |
| **Risk** | Infrastructure enumeration; exposes production endpoint; hardcoded config breaks on instance restart |
| **Action** | TARGET `utils.py` uses `API_URL = os.environ.get("API_URL", "http://localhost:8000")` |

---

### CRIT-05: Sensitive Data in Documentation
| | |
|---|---|
| **File** | `deployment_guide.md` (SOURCE) |
| **Exposed values** | EC2 IP, Instance ID, RDS hostname, RDS password, DockerHub username |
| **Risk** | Cloud infrastructure fully enumerable from documentation |
| **Action** | TARGET `deployment_guide.md` uses `<placeholder>` variables for all sensitive values |

---

## Medium Findings (RESOLVED)

### MED-01: No MIME Type Validation on File Upload (API)
| | |
|---|---|
| **File** | `api/main.py` (SOURCE) |
| **Issue** | Any file type could be uploaded to `/predict` endpoint |
| **Action** | TARGET `api/main.py` (merged) validates `content_type.startswith("image/")` — returns HTTP 415 for non-image uploads |

### MED-02: DB Session Leaks
| | |
|---|---|
| **File** | `api/main.py` (SOURCE) `/get_logs` route |
| **Issue** | `db.close()` only called in happy path, leaked on exception |
| **Action** | All DB sessions in TARGET wrapped in `try/finally db.close()` |

### MED-03: Unrestricted CORS
| | |
|---|---|
| **File** | `api/main.py` (TARGET, merged) |
| **Issue** | `allow_origins=["*"]` permits cross-origin requests from any domain |
| **Recommendation** | In production, restrict to specific origins: `allow_origins=["http://<EC2_IP>:8501"]` |
| **Status** | Accepted for research/demo; flagged for production hardening |

---

## Low Findings (INFO)

### LOW-01: Python Cache Files in Source Repository
| | |
|---|---|
| **File** | `experiments/__pycache__/`, `model/__pycache__/` (SOURCE) |
| **Issue** | Compiled `.pyc` files committed; can reveal bytecode |
| **Action** | `__pycache__/` already in `.gitignore`; not transferred to TARGET |

### LOW-02: Large Binary Model Checkpoint Committed
| | |
|---|---|
| **File** | `model/checkpoints/baseline_model.pth` (SOURCE, 13.5 MB) |
| **Issue** | Large binary bloats repository; not semantically meaningful in git history |
| **Action** | `*.pth` added to TARGET `.gitignore`; checkpoint not transferred |
| **Recommendation** | Store model artifacts in S3 or a model registry (MLflow, DVC) |

---

## Security Controls Now in Place

| Control | Implementation |
|---------|---------------|
| No hardcoded credentials | All secrets read from `DATABASE_URL` / `API_URL` env vars |
| `.env.example` | Provides template without real values |
| `.gitignore` additions | `*.pem`, `*.key`, `*.pth`, `.env` all blocked |
| MIME type validation | `/predict` rejects non-image uploads (HTTP 415) |
| DB session safety | `try/finally db.close()` in all routes |
| No sensitive data in docs | All IPs/passwords replaced with placeholders |

---

## Recommended Next Steps

1. **Rotate RDS password** — the old password `V3lvet#T0rnado` may be in any git history or CI logs
2. **Rotate EC2 key pair** — `Cloud26.pem` may be in git history
3. **Restrict RDS public access** — if only accessed from EC2, disable "Publicly accessible"
4. **Restrict CORS origins** — change `allow_origins=["*"]` to the actual UI origin in production
5. **Enable RDS encryption at rest** — ensure KMS-managed encryption is enabled
6. **Store model checkpoints in S3** — use `boto3` + presigned URLs rather than git
7. **Add API authentication** — protect MLOps routes with API key or JWT for production
