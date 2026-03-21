import cv2
import argparse
from pathlib import Path
from inference import WeaponDetector

def main():
    parser = argparse.ArgumentParser(description="Evaluate single image with the trained model.")
    parser.add_argument("image_path", type=str, help="Path to the image to evaluate")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image {img_path} not found.")
        return

    print("Loading model...")
    detector = WeaponDetector()
    
    print(f"Running inference on {img_path}...")
    result = detector.predict(str(img_path), conf_threshold=args.conf)
    
    # Save visualized result
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"result_{img_path.name}"
    
    # Plot plots the image array and returns it
    labeled_img = result.plot()
    
    # Save the array to disk
    cv2.imwrite(str(out_path), labeled_img)
    
    print(f"\n--- Detection Results ---")
    formatted = detector.format_predictions(result)
    print(f"Found {formatted['num_detections']} objects:")
    for det in formatted["detections"]:
        print(f"  - {det['class_name']} ({det['confidence']:.2f}) at {det['bbox']}")
        
    print(f"\nSaved visualization to {out_path}")

if __name__ == "__main__":
    main()
