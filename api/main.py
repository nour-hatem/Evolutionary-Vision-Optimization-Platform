"""
api/main.py
-----------
FastAPI application for CIFAR-10 image classification.

Run with:
    uvicorn api.main:app --reload

Endpoint:
    POST /predict
        Accepts a multipart image upload and returns the predicted class + confidence.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.predict import load_model, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── lifespan: load model once at startup ──────────────────────────────────────

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


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description=(
        "Production inference API backed by a GA-optimised CNN trained on CIFAR-10. "
        "Upload any image and receive the predicted class label and confidence score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe — confirms the service is running."""
    return {"status": "ok"}


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(..., description="Image file to classify")) -> JSONResponse:
    """
    Classify an uploaded image.

    Accepts any image format supported by Pillow (JPEG, PNG, BMP, WEBP, …).

    Returns
    -------
    JSON
        {
            "prediction": "<cifar10 class>",
            "confidence": <float 0-1>
        }
    """
    # ── validate MIME type ─────────────────────────────────────────────────────
    allowed_mime_prefixes = ("image/",)
    content_type = (file.content_type or "").lower()
    if not any(content_type.startswith(p) for p in allowed_mime_prefixes):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload an image file.",
        )

    # ── read payload ───────────────────────────────────────────────────────────
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── run inference ──────────────────────────────────────────────────────────
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

    return JSONResponse(content=result)
