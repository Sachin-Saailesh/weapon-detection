# Edge AI Vision Security Platform

**Enterprise Threat Detection System · ONNX Accelerated · Multi-Node Support**

This repository contains a production-grade Edge AI pipeline engineered for identifying firearms in degraded CCTV feeds. Originally conceptualized as a YOLOv8 vs RT-DETR benchmark, the system has been architected into a resilient microservice capable of robust, asynchronous video analysis for real-world security deployments.

## 🚀 Key Architectural Features

### 1. Model Precision & Custom Filtering
- **Aggressive Hard-Negative Mining:** Trained on 35,000+ CCTV artifacts with thousands of negative samples to practically eliminate false-positive alerts on common objects.
- **Geometric Heuristics & NMS:** Raw inference logic is fortified with strict Non-Maximum Suppression (IoU 0.45) and a spatial heuristic filter that automatically rejects bounding boxes occupying >25% of the frame (preventing edge-cases like the model misclassifying a mounted CCTV camera body as a weapon).

### 2. Edge-Optimized Deployment
- **ONNX Inference:** Automatically converts the PyTorch `.pt` model into an `.onnx` graph on first execution, reducing VRAM usage and accelerating inference latency by up to 40% on CPU/Edge hardware.
- **Failover Logic:** Designed with an automatic fallback to PyTorch if the ONNX runtime fails to initialize.

### 3. Asynchronous Streaming & Resilience
- **FastAPI Backend:** Provides non-blocking API endpoints (`/predict`) and continuous MJPEG stream yielding (`/stream/{node_id}`).
- **Exponential Backoff Reconnect:** Security cameras routinely drop packets. The pipeline features an asynchronous reconnect loop that automatically catches feed drops, waits, and reconnects without crashing the container or DDOSing the camera network.

### 4. Telemetry & Observability
- **Prometheus Integration:** Exposes a `/metrics` endpoint tracking `inference_latency_seconds` (histogram), `active_streams` (gauge), and `frames_processed_total` (counter) for seamless Grafana dashboarding.
- **Unified UI/API Assembly:** Operates via a single ASGI application where the REST endpoints and a professional Gradio dashboard coexist on the exact same port.

---

## 💻 Installation & Usage

### Local Setup
Ensure you are running Python 3.10+ (Tested extensively on Python 3.13 macOS and 3.11 Linux).

```bash
# Create internal environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Starting the Server
The entire platform (API + Dashboard) boots up with a single ASGI command:
```bash
python gradio_app.py
```
*Alternatively, you can run `uvicorn api.main:app --host 0.0.0.0 --port 8000` to expose the backend APIs concurrently with the UI.*

---

## 📡 API Reference
When hosted via FastAPI (e.g. Hugging Face Spaces):
- `/` — Web Dashboard (Gradio Enterprise UI)
- `/health` — Diagnostics & Hardware verification
- `/metrics` — Prometheus structured telemetry
- `/cameras` — Matrix of currently registered visual nodes
- `/stream/{camera_id}` — Continuous MJPEG artifact inference stream

## 📝 Background Research
For a deeper dive into the transition from a university computer vision benchmark to a production-level architectural system, read the included `info.md` whitepaper.
