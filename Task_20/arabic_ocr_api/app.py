"""
FastAPI application for the Arabic OCR (CRNN + CTC) microservice.

Endpoints:
    GET  /         - health check
    POST /predict  - upload an image, get back recognized Arabic text
"""

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ocr.decoding import decode_predictions_beam
from ocr.model_loader import ocr_model
from ocr.preprocessing import preprocess_image_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr.app")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the TensorFlow model, vocab, and config exactly once at startup,
    # not per-request.
    logger.info("Starting up: loading OCR model...")
    ocr_model.load()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Arabic OCR Microservice",
    description="CRNN + CTC based Arabic text recognition for national ID address images.",
    version="1.0.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocab_size: int | None = None
    ctc_time_steps: int | None = None


class PredictResponse(BaseModel):
    success: bool
    filename: str
    recognized_text: str


@app.get("/", response_model=HealthResponse)
def health_check():
    """Health check endpoint reporting whether the model is loaded and ready."""
    model_loaded = ocr_model.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "starting",
        model_loaded=model_loaded,
        vocab_size=ocr_model.config["VOCAB_SIZE"] if ocr_model.config else None,
        ctc_time_steps=ocr_model.config["CTC_TIME_STEPS"] if ocr_model.config else None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Accepts an image file, preprocesses it, runs inference, and returns
    the recognized Arabic text."""
    if ocr_model.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed types: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        img_h = ocr_model.config["IMG_HEIGHT"]
        img_w = ocr_model.config["IMG_WIDTH"]
        image_tensor = preprocess_image_bytes(image_bytes, img_h, img_w)
    except Exception as exc:
        logger.exception("Failed to preprocess uploaded image")
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    batch = np.expand_dims(image_tensor.numpy(), axis=0)  # (1, W, H, 1)

    try:
        preds = ocr_model.model.predict(batch, verbose=0)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    texts = decode_predictions_beam(preds, ocr_model.idx_to_char, beam_width=10)

    return PredictResponse(
        success=True,
        filename=file.filename or "unknown",
        recognized_text=texts[0],
    )
