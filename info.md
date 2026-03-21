# Edge AI Vision Security Platform: From Coursework to Production Pipeline

## Abstract
What began as an "Intro to Computer Vision" university project benchmarking YOLOv11 against RT-DETR has evolved into a fully-fledged Edge AI Vision Security Platform. While initially focused purely on algorithmic precision, real-world deployment quickly revealed that *model accuracy* is only a fraction of the equation. This paper outlines the architectural pivot from simple model inference toward robust systems engineering—emphasizing ONNX acceleration, asynchronous streaming, fault-tolerant retry logic, and comprehensive observability.

---

## 1. The Journey: Course Project to Systems Engineering
### 1.1 The Genesis
The initial scope of this project was an academic benchmark. The goal was to compare the performance of YOLOv11 and RT-DETR for detecting weapons in degraded CCTV footage. We leveraged the [Hyper.ai Gun Detection Dataset](https://hyper.ai/en/datasets/20876), which provides 51,000 annotated images designed for analyzing real-time weapon detection in surveillance videos. We augmented this foundation with rigorous hard-negative mining and plotted P-R curves. The YOLO-based architecture ultimately won out due to its superior inference speed on edge hardware without sacrificing mAP on small objects.

### 1.2 The "AI Engineer" Pivot
Building a Jupyter Notebook that detects weapons in static images is a data science task. Building a system that can ingest 24/7 RTSP streams, recover from network drops, and serve low-latency detections via API is an **AI Engineering** task. 

The transition required decoupling the model from the script and embedding it within a resilient pipeline. The focus shifted from *epochs* and *learning rates* to *latency*, *fallback strategies*, *retry logics*, and *deployment metrics*.

---

## 2. Platform Architecture & Capabilities

### 2.1 ONNX Acceleration & Inference Optimization
Deploying PyTorch weights (`.pt`) directly in production introduces unnecessary overhead. To optimize for edge hardware (like NVIDIA Jetson or CPU-only servers):
- **Dynamic Export**: The system automatically exports the PyTorch model to an ONNX graph (`best.onnx`) upon first execution.
- **Inference Engine**: We migrated inference to `onnxruntime`, reducing inference latency by up to 40% compared to native PyTorch, achieving a steady state of ~60-80ms per frame even on hardware without dedicated GPUs.
- **NMS & Geometric Heuristics**: Raw ONNX outputs expose thousands of anchor boxes. We implemented rigorous Non-Maximum Suppression (NMS, IoU=0.45) layered with **geometric heuristic filters** (rejecting bounding boxes >25% of frame area) to surgically eliminate false positives like mounted CCTV cameras being misclassified as weapons.

### 2.2 Asynchronous Streaming & Concurrency
Real-world security systems rely on continuous video feeds (RTSP/HTTP MJPEG), not static image uploads.
- **FastAPI Integration**: The platform exposes `/stream/{camera_id}`, leveraging Python's `asyncio` to yield MJPEG frames continuously without blocking the main thread.
- **Multi-Camera Registry**: A centralized registry abstracts camera sources, allowing the system to handle multiple streams concurrently. 

### 2.3 Fault Tolerance & Retry Logic
Security systems must be highly resilient. If a camera feed drops, the pipeline cannot simply crash or exit.
- **Stream Reconnection Engine**: Implemented in the MJPEG generator, the pipeline catches `cv2.VideoCapture` read failures.
- **Exponential Backoff**: Upon feed loss, the system triggers an automatic reconnect loop with exponential backoff (starting at 1.0s, scaling up to 30.0s), preventing DDOS-like behavior on the camera network while guaranteeing immediate recovery when the stream returns.
- **Inference Fallback**: The `WeaponDetector` class is designed with graceful degradation. If the ONNX runtime fails to initialize or the `.onnx` graph is corrupted, it automatically falls back to native PyTorch inference, ensuring zero downtime.

### 2.4 Observability & Telemetry
"If you can't measure it, you can't manage it." To track system health, we integrated **Prometheus**:
- `inference_latency_seconds` (Histogram): Tracks the microsecond latency of the ONNX execution.
- `weapons_detected_total` (Counter): Logs aggregate security events over time.
- `frames_processed_total` (Counter): Tracks pipeline throughput per camera.
- `active_streams` (Gauge): Monitors the current load on the API.

These metrics allow for seamless Grafana dashboarding, giving system administrators a live view of platform health rather than relying on standard out logs.

---

## 3. LinkedIn Post Draft

*(Drafted to be story-driven, highlighting the transition from academic benchmark to engineering system)*

**Title Idea:** Stop Building Projects. Start Building Systems. 🚀

**Content:**
What started as my "Intro to Computer Vision" coursework just turned into a production-grade Edge AI Security Platform. 

Initially, my goal was simple: benchmark YOLOv11 vs RT-DETR for weapon detection on low-quality CCTV feeds. After fine-tuning on the [Hyper.ai Gun Detection Dataset](https://hyper.ai/en/datasets/20876) (51,000+ CCTV artifacts) coupled with heavy hard-negative mining, I had a great model. But I quickly realized a hard truth—a highly accurate Jupyter notebook does absolutely nothing in the real world. 

To bridge the gap between Data Science and AI Engineering, I entirely rebuilt the deployment pipeline. Here’s what it took to get this application "edge-ready":

⚡️ **ONNX Acceleration:** Stripped away PyTorch overhead, utilizing `onnxruntime` to slash inference latency and allow the model to run on lightweight edge hardware.
🔄 **Resilient Reconnects:** Security cameras drop packets constantly. I engineered an asynchronous streaming backend (FastAPI) with exponential backoff retry logic. If a feed drops, the system waits, recovers, and continues without manual reboots.
🛡 **Algorithmic Guardrails:** Raw logic isn't enough. I layered Non-Maximum Suppression (NMS) with custom geometric heuristic filters to eliminate ridiculous false positives (like the model mistaking a massive mounted camera for a handgun).
📊 **Observability:** Integrated Prometheus telemetry to track `inference_latency_seconds`, active stream counts, and throughput in real-time. 

Building ML models is fun. But engineering the fault-tolerant, asynchronous, monitored systems that those models live inside? That’s where the real challenge is. 

Check out the architecture breakdown and deployment below! 👇 

*(Include a video/GIF of the Gradio Live Stream dashboard or the GitHub repository link here)*

#AIEngineering #ComputerVision #YOLOv8 #ONNX #FastAPI #SystemArchitecture #EdgeAI
