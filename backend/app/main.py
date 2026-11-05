import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

logger = logging.getLogger("roadvision")
logging.basicConfig(level=logging.INFO)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/yolo11n_v2.pt"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
MAX_BYTES = 10 * 1024 * 1024
model = None
model_lock = Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH.resolve()}")
    model = YOLO(str(MODEL_PATH))
    logger.info("Loaded model_version=%s path=%s", MODEL_VERSION, MODEL_PATH)
    yield
    model = None

app = FastAPI(title="RoadVision API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "unavailable", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(file: UploadFile = File(...), conf: float = 0.15):
    if not 0.01 <= conf <= 1.0:
        raise HTTPException(422, "conf must be between 0.01 and 1.0")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Only JPEG, PNG, and WebP images are allowed")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 10 MB limit")
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img.load()
        if pil_img.width * pil_img.height > 20_000_000:
            raise HTTPException(413, "Image dimensions too large")
        pil_img = pil_img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid image file")
    begin = time.perf_counter()
    with model_lock:
        result = model.predict(source=np.array(pil_img), conf=conf, imgsz=640, verbose=False)[0]
    detections = []
    for box in result.boxes:
        cls = int(box.cls[0].item())
        detections.append({
            "class_id": cls,
            "class_name": result.names[cls],
            "confidence": round(float(box.conf[0].item()), 4),
            "xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
        })
    latency_ms = round((time.perf_counter() - begin) * 1000, 2)
    logger.info("predict model_version=%s detections=%d latency_ms=%.2f", MODEL_VERSION, len(detections), latency_ms)
    return {
        "model_version": MODEL_VERSION,
        "image_width": pil_img.width,
        "image_height": pil_img.height,
        "threshold": conf,
        "latency_ms": latency_ms,
        "detections": detections,
    }
