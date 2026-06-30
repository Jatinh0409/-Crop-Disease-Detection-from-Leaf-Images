"""
app.py
FastAPI service for crop disease detection.

Run locally:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /health             -> liveness check
    POST /predict             -> upload an image, get class + confidence
    POST /predict_with_gradcam -> same, plus a base64 Grad-CAM heatmap overlay
    GET  /metrics              -> last-recorded training metrics
"""

import base64
import io
import json
import os
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from gradcam import load_model_for_gradcam, predict_with_gradcam  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "../models/model.pt")
METRICS_PATH = os.getenv("METRICS_PATH", "../models/metrics.json")

app = FastAPI(
    title="Crop Disease Detection API",
    description="Leaf-image disease classification with Grad-CAM explainability",
    version="1.0.0",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = None


@app.on_event("startup")
def load_model():
    global model, class_names
    model, class_names = load_model_for_gradcam(MODEL_PATH, device)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_class_probs: dict
    latency_ms: float


class GradCamResponse(PredictionResponse):
    gradcam_overlay_base64: str


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "classes": class_names}


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="No metrics file found yet — train the model first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    image = _read_image(image_bytes)

    start = time.perf_counter()
    result = predict_with_gradcam(image, model, class_names, device)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=round(result["confidence"], 4),
        all_class_probs={k: round(v, 4) for k, v in result["all_class_probs"].items()},
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict_with_gradcam", response_model=GradCamResponse)
async def predict_with_gradcam_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    image = _read_image(image_bytes)

    start = time.perf_counter()
    result = predict_with_gradcam(image, model, class_names, device)
    latency_ms = (time.perf_counter() - start) * 1000

    buf = io.BytesIO()
    result["overlay_image"].save(buf, format="JPEG", quality=90)
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return GradCamResponse(
        predicted_class=result["predicted_class"],
        confidence=round(result["confidence"], 4),
        all_class_probs={k: round(v, 4) for k, v in result["all_class_probs"].items()},
        latency_ms=round(latency_ms, 2),
        gradcam_overlay_base64=overlay_b64,
    )
