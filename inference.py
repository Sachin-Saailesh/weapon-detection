import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Fix for PyTorch 2.6+ strict weight loading
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load


class WeaponDetector:
    """
    YOLOv8 weapon detector supporting both PyTorch and ONNX inference.
    Automatically exports ONNX model on first load if not present.
    """

    def __init__(self, model_path: str = "models/best.pt", use_onnx: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        self.pt_path = model_path
        self.onnx_path = str(Path(model_path).with_suffix(".onnx"))
        self.use_onnx = use_onnx

        # Always load PyTorch model (needed for class names and export)
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"[Inference] Loaded PyTorch model from {model_path}")

        # Export to ONNX if requested and not already done
        if use_onnx:
            if not os.path.exists(self.onnx_path):
                print(f"[Inference] Exporting ONNX model to {self.onnx_path} ...")
                self.model.export(format="onnx", imgsz=640, opset=12, simplify=True)
                print(f"[Inference] ONNX export complete.")

            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.ort_session.get_inputs()[0].name
                print(f"[Inference] ONNX runtime loaded. Using ONNX for inference.")
            except ImportError:
                print("[Inference] onnxruntime not installed — falling back to PyTorch.")
                self.use_onnx = False
                self.ort_session = None
        else:
            self.ort_session = None

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(self, image_source, conf_threshold: float = 0.40):
        """Run inference on a single image (numpy array, PIL Image, or path)."""
        results = self.model.predict(
            source=image_source,
            conf=conf_threshold,
            device="cpu",
            verbose=False
        )
        return results[0]

    def predict_onnx(self, frame_bgr: np.ndarray, conf_threshold: float = 0.50):
        """
        ONNX-optimised inference on a single BGR frame (numpy array).
        Applies NMS to suppress duplicate boxes.
        Returns same dict format as format_predictions, plus annotated frame.
        """
        if self.ort_session is None:
            result = self.predict(frame_bgr, conf_threshold)
            return self.format_predictions(result), result.plot()

        # Pre-process
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # NCHW

        # Run ONNX
        outputs = self.ort_session.run(None, {self.input_name: img})
        # YOLOv8 ONNX output shape: [1, 4+num_classes, 8400] → transpose to [8400, 4+num_classes]
        preds = outputs[0][0].T  # shape: [8400, 4+num_classes]

        h, w = frame_bgr.shape[:2]

        # --- Collect raw candidates ---
        raw_boxes   = []   # [x1, y1, x2, y2] in original frame coords
        raw_scores  = []   # best class confidence
        raw_clsids  = []   # class id for that best score

        for pred in preds:
            scores = pred[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < conf_threshold:
                continue
            cx, cy, bw, bh = pred[:4]
            x1 = max(0, int((cx - bw / 2) / 640 * w))
            y1 = max(0, int((cy - bh / 2) / 640 * h))
            x2 = min(w, int((cx + bw / 2) / 640 * w))
            y2 = min(h, int((cy + bh / 2) / 640 * h))

            raw_boxes.append([x1, y1, x2 - x1, y2 - y1])   # xywh for NMS
            raw_scores.append(conf)
            raw_clsids.append(cls_id)

        # --- Apply NMS ---
        nms_detections = []
        if raw_boxes:
            indices = cv2.dnn.NMSBoxes(
                raw_boxes,
                raw_scores,
                score_threshold=conf_threshold,
                nms_threshold=0.45
            )
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw2, bh2 = raw_boxes[i]
                    nms_detections.append({
                        "bbox": [x, y, x + bw2, y + bh2],
                        "confidence": round(raw_scores[i], 4),
                        "class_id": raw_clsids[i],
                        "class_name": self.class_names.get(raw_clsids[i], f"class_{raw_clsids[i]}")
                    })

        # --- Geometric heuristic filter ---
        # Reject bbox > 25% of frame area (handheld guns < 25% of frame)
        # Also reject degenerate tiny boxes < 8px
        image_area = h * w
        detections = []
        for det in nms_detections:
            x1, y1, x2, y2 = det["bbox"]
            box_w, box_h = x2 - x1, y2 - y1
            if (box_w * box_h) / image_area > 0.25:
                continue
            if box_w < 8 or box_h < 8:
                continue
            detections.append(det)

        # Draw bounding boxes for visualization
        annotated = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return {
            "num_detections": len(detections),
            "detections": detections,
            "image_shape": (h, w)
        }, annotated

    # ------------------------------------------------------------------
    # Video stream generator
    # ------------------------------------------------------------------

    def stream_predict(self, source, conf_threshold: float = 0.40):
        """
        Generator that yields (formatted_result, annotated_frame, latency_ms) tuples
        for every frame in a video source.

        Args:
            source: int (camera index), str (file path or RTSP/HTTP URL)
            conf_threshold: detection confidence threshold
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"[Stream] Cannot open video source: {source}")

        print(f"[Stream] Opened source: {source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                formatted, annotated = self.predict_onnx(frame, conf_threshold)
                latency_ms = (time.perf_counter() - t0) * 1000

                yield formatted, annotated, latency_ms
        finally:
            cap.release()
            print(f"[Stream] Closed source: {source}")

    # ------------------------------------------------------------------
    # Formatting helper
    # ------------------------------------------------------------------

    def format_predictions(self, result):
        """Format ultralytics Results object for API consumption."""
        detections = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.40:
                continue
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": self.model.names[cls_id]
            })
        return {
            "num_detections": len(detections),
            "detections": detections,
            "image_shape": result.orig_shape
        }


if __name__ == "__main__":
    print("Testing inference module...")
    detector = WeaponDetector()
    print("Available classes:", detector.class_names)
    print("ONNX mode:", detector.use_onnx)
