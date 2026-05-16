"""
api/main.py
-----------
FastAPI application for CIFAR-10 image classification AND MLOps Experiment Tracking.

Run with:
    uvicorn api.main:app --reload

Endpoints:
    GET  /           — service info
    GET  /health     — liveness probe (includes DB status)
    POST /predict    — upload an image and get predicted class + confidence
    POST /start_experiment   — trigger a GA optimization run (logged to AWS RDS)
    POST /resume_experiment  — resume an interrupted experiment from checkpoint
    GET  /get_logs/{experiment_id}       — fetch per-generation metrics from RDS
    GET  /experiment_status/{run_name}   — full experiment status + checkpoint info
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.predict import load_model, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Experiment manager is optional — the API boots fine without a DB connection so
# that /health and /predict still work in environments without RDS access.
_experiment_manager = None


def _get_manager():
    global _experiment_manager
    if _experiment_manager is None:
        try:
            from experiments.manager import ExperimentManager
            _experiment_manager = ExperimentManager()
        except Exception as exc:
            logger.warning("ExperimentManager unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Experiment tracking service is unavailable (check DATABASE_URL).",
            )
    return _experiment_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model …")
    try:
        load_model()
        logger.info("Model ready.")
    except FileNotFoundError as exc:
        logger.error("Startup failed — %s", exc)
        # Allow the app to start even without a checkpoint so /health still works.
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="CIFAR-10 Platform",
    description=(
        "Inference API and MLOps Experiment Tracking connected to AWS RDS. "
        "Upload any image and receive the predicted class label and confidence score. "
        "Use the /start_experiment and /resume_experiment routes to manage GA runs."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    run_name: str
    max_generations: int
    population_size: int
    mutation_rate: float
    dataset: str
    interrupt_after: Optional[int] = None


class ResumeConfig(BaseModel):
    run_name: str


# ── Ops routes ────────────────────────────────────────────────────────────────

@app.get("/", tags=["ops"])
def root() -> dict:
    return {
        "service": "CIFAR-10 Platform",
        "version": "2.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe — confirms the service is running."""
    db_status = "unknown"
    try:
        from experiments.db_models import engine
        with engine.connect():
            db_status = "connected"
    except Exception:
        db_status = "unavailable"
    return {"status": "ok", "database": db_status}


# ── Inference route ────────────────────────────────────────────────────────────

@app.post("/predict", tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Image file to classify")
) -> JSONResponse:
    """
    Classify an uploaded image.

    Accepts any image format supported by Pillow (JPEG, PNG, BMP, WEBP, …).

    Returns
    -------
    JSON
        {"prediction": "<cifar10 class>", "confidence": <float 0-1>}
    """
    # Validate MIME type
    allowed_mime_prefixes = ("image/",)
    content_type = (file.content_type or "").lower()
    if not any(content_type.startswith(p) for p in allowed_mime_prefixes):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload an image file.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_single(raw_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference error for file '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    logger.info(
        "Predicted '%s' (%.2f%%) for file '%s'",
        result["prediction"],
        result["confidence"] * 100,
        file.filename,
    )

    # Log prediction to database (non-fatal if DB is down)
    try:
        from experiments.db_models import SessionLocal, PredictionLog
        db = SessionLocal()
        log = PredictionLog(
            filename=file.filename,
            predicted_class=result.get("prediction", "unknown"),
            confidence=result.get("confidence", 0.0),
        )
        db.add(log)
        db.commit()
        db.close()
    except Exception as e:
        logger.warning("Failed to log prediction to DB: %s", e)

    return JSONResponse(content=result)


# ── MLOps routes ──────────────────────────────────────────────────────────────

@app.post("/start_experiment", tags=["mlops"])
async def start_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """
    Trigger the GA optimization loop and log results to AWS RDS.
    Set interrupt_after to simulate a Spot instance interruption after N generations.
    """
    manager = _get_manager()
    config_dict = config.model_dump()
    interrupt_after = config_dict.pop("interrupt_after", None)
    background_tasks.add_task(manager.run, config.run_name, config_dict, interrupt_after)

    msg = f"Experiment '{config.run_name}' started"
    if interrupt_after:
        msg += f" (will simulate interruption after {interrupt_after} generations)"
    msg += " — logging to AWS RDS!"

    return {"status": "success", "message": msg}


@app.post("/resume_experiment", tags=["mlops"])
async def resume_experiment(config: ResumeConfig, background_tasks: BackgroundTasks):
    """Resume an interrupted experiment from its last checkpoint in RDS."""
    manager = _get_manager()

    from experiments.db_models import SessionLocal, Experiment as ExpModel
    db = SessionLocal()
    try:
        exp = (
            db.query(ExpModel)
            .filter(
                ExpModel.run_name == config.run_name,
                ExpModel.status == "interrupted",
            )
            .first()
        )
        if not exp:
            raise HTTPException(
                status_code=404,
                detail=f"No interrupted experiment found with name '{config.run_name}'",
            )
    finally:
        db.close()

    background_tasks.add_task(manager.resume, config.run_name)
    return {
        "status":  "success",
        "message": f"Experiment '{config.run_name}' is resuming from checkpoint!",
    }


@app.get("/get_logs/{experiment_id}", tags=["mlops"])
async def get_logs(experiment_id: int):
    """Fetch real-time per-generation training logs from AWS RDS."""
    from experiments.db_models import SessionLocal, RunLog
    db = SessionLocal()
    try:
        logs = db.query(RunLog).filter(RunLog.experiment_id == experiment_id).all()
        return [
            {"generation": l.generation, "accuracy": l.accuracy, "f1": l.f1_score}
            for l in logs
        ]
    finally:
        db.close()


@app.get("/experiment_status/{run_name}", tags=["mlops"])
async def experiment_status(run_name: str):
    """Check the status of an experiment including checkpoint info."""
    from experiments.db_models import SessionLocal, Experiment as ExpModel, RunLog, Checkpoint
    db = SessionLocal()
    try:
        exp = db.query(ExpModel).filter(ExpModel.run_name == run_name).first()
        if not exp:
            raise HTTPException(status_code=404, detail=f"Experiment '{run_name}' not found")

        logs = db.query(RunLog).filter(RunLog.experiment_id == exp.id).all()
        ckpt = db.query(Checkpoint).filter(Checkpoint.experiment_id == exp.id).first()

        return {
            "experiment_id":         exp.id,
            "run_name":              exp.run_name,
            "status":                exp.status,
            "config":                exp.config,
            "start_time":            str(exp.start_time),
            "end_time":              str(exp.end_time) if exp.end_time else None,
            "runtime_seconds":       exp.runtime_seconds,
            "generations_completed": len(logs),
            "logs": [
                {"gen": l.generation, "accuracy": l.accuracy, "f1": l.f1_score}
                for l in logs
            ],
            "checkpoint": {
                "generation": ckpt.generation,
                "state":      ckpt.population_state,
            } if ckpt else None,
        }
    finally:
        db.close()
