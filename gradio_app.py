import gradio as gr
import cv2
import numpy as np
import tempfile
import time
import os
from inference import WeaponDetector

# Initialize the detector (ONNX mode enabled)
detector = WeaponDetector(model_path="models/best.pt", use_onnx=True)


# ------------------------------------------------------------------
# Tab 1: Single Image Detection
# ------------------------------------------------------------------
def detect_image(image, conf_threshold):
    if image is None:
        return None, "Please upload an image."

    # image comes in as numpy RGB from Gradio
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    formatted, annotated_bgr = detector.predict_onnx(frame_bgr, conf_threshold=conf_threshold)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Convert annotated BGR back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    n = formatted["num_detections"]
    mode = "ONNX Runtime" if detector.use_onnx else "PyTorch"
    summary = f"**Execution Mode:** {mode} | **Inference Latency:** {latency_ms:.1f} ms | **Total Detections:** {n}\n\n"
    if n == 0:
        summary += "**STATUS: SECURE** — No threat anomalies detected."
    else:
        summary += f"**STATUS: CRITICAL** — {n} threat anomaly(s) identified:\n"
        for det in formatted["detections"]:
            summary += f"- **Classification:** {det['class_name']} | **Confidence:** {det['confidence']:.2f} | **Bounding Box:** {det['bbox']}\n"

    return annotated_rgb, summary


# ------------------------------------------------------------------
# Tab 2: Video File Detection
# ------------------------------------------------------------------
def detect_video(video_path, conf_threshold, progress=gr.Progress()):
    if video_path is None:
        return None, "Please upload a video file."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to a temp file
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total_weapons = 0
    total_latency = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        formatted, annotated = detector.predict_onnx(frame, conf_threshold=conf_threshold)
        latency_ms = (time.perf_counter() - t0) * 1000

        total_weapons += formatted["num_detections"]
        total_latency += latency_ms
        frame_idx += 1

        # Per-frame HUD
        hud = f"Frame {frame_idx}/{total_frames} | Threats: {formatted['num_detections']} | {latency_ms:.0f}ms"
        cv2.putText(annotated, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        writer.write(annotated)
        if total_frames > 0:
            progress(frame_idx / total_frames)

    cap.release()
    writer.release()

    avg_latency = total_latency / max(frame_idx, 1)
    mode = "ONNX Runtime" if detector.use_onnx else "PyTorch"
    summary = (
        f"### Batch Processing Metrics\n"
        f"- **Execution Engine:** {mode}\n"
        f"- **Frames Validated:** {frame_idx}\n"
        f"- **Avg. Inference Latency:** {avg_latency:.1f} ms/frame\n"
        f"- **Aggregate Threat Detections:** {total_weapons}\n"
    )
    return out_path, summary


# ------------------------------------------------------------------
# Tab 3: Live RTSP / Camera Stream (snapshot mode)
# ------------------------------------------------------------------
def capture_stream_snapshot(camera_source, conf_threshold):
    """
    Grabs a single frame from a camera index or RTSP URL, runs detection,
    and returns the annotated frame. Intended to be called repeatedly for
    a live-ish feed via Gradio's polling mechanism.
    """
    try:
        source = int(camera_source) if camera_source.strip().isdigit() else camera_source.strip()
    except Exception:
        source = camera_source.strip()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None, f"❌ Cannot open source: `{source}`. Check camera index or URL."

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "System Error: Failed to read frame from active source."

    t0 = time.perf_counter()
    formatted, annotated = detector.predict_onnx(frame, conf_threshold=conf_threshold)
    latency_ms = (time.perf_counter() - t0) * 1000

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    n = formatted["num_detections"]
    status_label = "CRITICAL" if n > 0 else "SECURE"
    status = (
        f"**STATUS: {status_label}** — {n} threat(s) detected. | "
        f"**Latency:** {latency_ms:.1f} ms | **Active Source:** `{source}`"
    )
    return annotated_rgb, status


# ------------------------------------------------------------------
# Build Gradio Interface
# ------------------------------------------------------------------
with gr.Blocks(
    title="Edge AI Vision Security Platform",
    theme=gr.themes.Monochrome()
) as demo:
    gr.Markdown("""
    # Edge AI Vision Security Platform
    **Enterprise Threat Detection System · ONNX Accelerated · Multi-Node Support**

    <div style="background-color: var(--block-background-fill); border: 1px solid var(--border-color-primary); padding: 10px 14px; border-radius: 6px; font-size: 0.85em; margin-top: 10px; margin-bottom: 10px;">
    Leveraging a YOLOv8-based architecture fine-tuned on +35K CCTV artifacts. Features aggressive hard-negative mining, asynchronous inference streams, and exponential backoff retry algorithms for edge hardware resilience.
    </div>
    """)

    with gr.Tabs():
        # ---- Tab 1: Image ----
        with gr.TabItem("Static Inference"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Input Source")
                    img_conf = gr.Slider(0.1, 1.0, value=0.50, step=0.05, label="Confidence Threshold (α)")
                    img_btn = gr.Button("Execute Inference", variant="primary")
                with gr.Column():
                    img_output = gr.Image(type="numpy", label="Processed Output")
                    img_summary = gr.Markdown(label="Inference Telemetry")

            img_btn.click(detect_image, [img_input, img_conf], [img_output, img_summary])

        # ---- Tab 2: Video ----
        with gr.TabItem("Batch Video Processing"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Input Video Stream")
                    vid_conf = gr.Slider(0.1, 1.0, value=0.50, step=0.05, label="Confidence Threshold (α)")
                    vid_btn = gr.Button("Initialize Batch Processing", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Annotated Execution Output")
                    vid_summary = gr.Markdown(label="Batch Telemetry")

            vid_btn.click(detect_video, [vid_input, vid_conf], [vid_output, vid_summary])

        # ---- Tab 3: Live Stream ----
        with gr.TabItem("Live Edge Telemetry"):
            gr.Markdown("""
            **Diagnostic Feed Authentication**
            Enter an authorized hardware index (e.g., `0`) or remote RTSP endpoint. 
            Utilize **Capture Frame** for single-cycle diagnosis. 
            For continuous, unbuffered monitoring, interface with the asynchronous MJPEG API endpoint: `GET /stream/{camera_id}`
            """)
            with gr.Row():
                with gr.Column():
                    stream_src = gr.Textbox(
                        label="Source Identifier / RTSP URL",
                        placeholder="0 or rtsp://user:pass@ip:554/stream",
                        value="0"
                    )
                    stream_conf = gr.Slider(0.1, 1.0, value=0.50, step=0.05, label="Confidence Threshold (α)")
                    stream_btn = gr.Button("Execute Diagnostic Capture", variant="primary")
                with gr.Column():
                    stream_output = gr.Image(type="numpy", label="Inference Snapshot")
                    stream_status = gr.Markdown(label="System Status")

            stream_btn.click(
                capture_stream_snapshot,
                [stream_src, stream_conf],
                [stream_output, stream_status]
            )

    gr.Markdown("""
    ---
    ### Administrative API Endpoints
    Available while running via `uvicorn api.main:app`. Consult the documentation for rate limits and payload schemas.
    - `POST /predict` — Rest API for static inference
    - `GET /stream/{node_id}` — Asynchronous MJPEG continuous stream
    - `GET /cameras` — Active node registry
    - `GET /metrics` — Exposes Prometheus structured telemetry (frames_processed, anomalies_detected, latency)
    - `GET /health` — Edge diagnostic subsystem
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
