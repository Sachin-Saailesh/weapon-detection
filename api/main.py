import os
import io
import time
import asyncio
import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import uvicorn
import sys
import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import WeaponDetector

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

app = FastAPI(
    title="Edge AI Weapon Detection API",
    description="ONNX-optimised real-time weapon detection for multi-camera CCTV deployments.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Model — shared across all endpoints
# ------------------------------------------------------------------
detector = WeaponDetector(model_path="models/best.pt", use_onnx=True)

# ------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------
WEAPONS_DETECTED = Counter(
    "weapons_detected_total",
    "Total number of weapons detected",
    ["camera_id"]
)
INFERENCE_TIME = Histogram(
    "inference_latency_seconds",
    "Per-frame inference latency",
    ["camera_id", "mode"]
)
FRAMES_PROCESSED = Counter(
    "frames_processed_total",
    "Total frames processed per camera",
    ["camera_id"]
)
ACTIVE_STREAMS = Gauge(
    "active_streams",
    "Number of currently active video streams"
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ------------------------------------------------------------------
# Multi-camera registry
# Camera sources can be: 0 (webcam), RTSP URL, file path, HTTP stream
# Override by setting env var CAMERA_SOURCES as JSON, e.g.:
#   '{"cam0": 0, "cam1": "rtsp://..."}'
# ------------------------------------------------------------------
import json
_default_cameras = {"cam0": 0}
try:
    _env_cameras = os.environ.get("CAMERA_SOURCES", "")
    CAMERA_REGISTRY: dict = json.loads(_env_cameras) if _env_cameras else _default_cameras
except Exception:
    CAMERA_REGISTRY = _default_cameras


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------
class DetectionResponse(BaseModel):
    num_detections: int
    detections: list
    inference_time_ms: float
    camera_id: str = "image"
    mode: str = "onnx"


# ------------------------------------------------------------------
# Single image endpoint (existing, enhanced)
# ------------------------------------------------------------------
@app.post("/predict", response_model=DetectionResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.40, ge=0.01, le=1.0)
):
    """Run weapon detection on an uploaded image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        with INFERENCE_TIME.labels(camera_id="image", mode="onnx").time():
            formatted, _ = detector.predict_onnx(frame_bgr, conf_threshold=conf)
        latency_ms = (time.perf_counter() - t0) * 1000

        FRAMES_PROCESSED.labels(camera_id="image").inc()
        if formatted["num_detections"] > 0:
            WEAPONS_DETECTED.labels(camera_id="image").inc(formatted["num_detections"])

        return JSONResponse({
            "num_detections": formatted["num_detections"],
            "detections": formatted["detections"],
            "inference_time_ms": round(latency_ms, 2),
            "camera_id": "image",
            "mode": "onnx" if detector.use_onnx else "pytorch"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Async MJPEG streaming endpoint
# ------------------------------------------------------------------
async def _mjpeg_frame_generator(camera_id: str, conf: float):
    """Async generator yielding MJPEG-encoded frames with detections drawn."""
    source = CAMERA_REGISTRY.get(camera_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not registered. Available: {list(CAMERA_REGISTRY.keys())}"
        )

    ACTIVE_STREAMS.inc()
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            ACTIVE_STREAMS.dec()
            raise HTTPException(
                status_code=503,
                detail=f"Cannot open camera source initially: {source}. Stream may be offline."
            )

        max_reconnect_attempts = 100
        reconnect_delay = 1.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[Stream Error] Camera {camera_id} feed lost. Attempting reconnect...")
                cap.release()

                # Exponential backoff reconnect
                reconnected = False
                for attempt in range(max_reconnect_attempts):
                    await asyncio.sleep(reconnect_delay)
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            print(f"[Stream Recovered] Successfully reconnected to {camera_id} on attempt {attempt+1}")
                            reconnected = True
                            reconnect_delay = 1.0  # reset backoff
                            break
                    cap.release()
                    reconnect_delay = min(reconnect_delay * 1.5, 30.0)  # max 30s backoff
                    print(f"[Stream Retry] Failed to connect to {camera_id}. Retrying in {reconnect_delay:.1f}s...")

                if not reconnected:
                    print(f"[Stream Terminated] Could not recover camera {camera_id} after {max_reconnect_attempts} attempts.")
                    break

            t0 = time.perf_counter()
            formatted, annotated = detector.predict_onnx(frame, conf_threshold=conf)
            latency_ms = (time.perf_counter() - t0) * 1000

            FRAMES_PROCESSED.labels(camera_id=camera_id).inc()
            INFERENCE_TIME.labels(camera_id=camera_id, mode="onnx").observe(latency_ms / 1000)
            if formatted["num_detections"] > 0:
                WEAPONS_DETECTED.labels(camera_id=camera_id).inc(formatted["num_detections"])

            # Overlay latency and detection count on frame
            label = f"ID:{camera_id} | {formatted['num_detections']} detections | {latency_ms:.1f}ms"
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            # Yield control back to the event loop
            await asyncio.sleep(0)

        cap.release()
    finally:
        ACTIVE_STREAMS.dec()


@app.get("/stream/{camera_id}")
async def stream_video(
    camera_id: str,
    conf: float = Query(0.40, ge=0.01, le=1.0)
):
    """
    MJPEG stream endpoint for a registered camera.
    Open in browser or VLC: http://localhost:8000/stream/cam0
    """
    return StreamingResponse(
        _mjpeg_frame_generator(camera_id, conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/cameras")
def list_cameras():
    """List all registered camera IDs."""
    return {"cameras": list(CAMERA_REGISTRY.keys())}


@app.get("/health")
def health_check():
    """Health and diagnostics endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "inference_mode": "onnx" if detector.use_onnx else "pytorch",
        "onnx_model": os.path.exists(detector.onnx_path),
        "active_streams": ACTIVE_STREAMS._value.get(),
        "registered_cameras": list(CAMERA_REGISTRY.keys())
    }

# ------------------------------------------------------------------
# Mount Gradio UI on root
# ------------------------------------------------------------------
from gradio_app import demo
# FastAPI app needs to be wrapped by Gradio's ASGI app for root mounting
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    print("Starting Edge AI Weapon Detection Server on port 8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
