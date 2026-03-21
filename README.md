# Weapon Detection System

This repository contains a full YOLOv8-based weapon detection pipeline, designed for business applications targeting the Indian market.

## Project Overview
This system is built from the ground-up to handle high-accuracy detection of weapons (specifically firearms) in CCTV footage, prioritizing robust performance in challenging environments like low-light, blur, and partial occlusions common in security camera feeds.

## Data Processing
The pipeline includes:
* **Custom Kaggle Dataset Integration:** Directly processes images without duplicating data to save disk space.
* **Hard Negative Mining:** Injects over 10,000 negative samples (images without weapons) to drastically reduce false positive alerts.
* **CCTV Augmentation Strategies:** Implements HSV shifts, rotation, and mosaic augmentations to simulate real-world security camera degradation.

## Training Pipeline
We utilize an Ultralytics YOLOv8 architecture optimized for both precision and recall.
* **Early Stopping:** The system stops automatically after 50 epochs of no improvement, preventing overfitting. 
* **Exporting:** Automatically exports the trained PyTorch (`.pt`) model to ONNX format for rapid edge deployment.

## Evaluation
The evaluation script calculates:
* mAP@50 and mAP@50-95
* Precision (crucial for alert fatigue) and Recall (crucial for security)
* Automatically generates Precision-Recall curves.

## Deployment Readiness
The pipeline includes mocked HTTP endpoints using FastAPI for deploying the trained model directly as a microservice, with provisions for Prometheus metrics integration.
