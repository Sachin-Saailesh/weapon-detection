# Weapon Detection Model - Performance Report

## Overview
This report analyzes the performance of the newly trained YOLOv8s weapon detection model. The model was initially configured and trained in Kaggle for 300 epochs (though metadata shows 5 epochs, likely due to early-stopping). The model has been extracted from the output zip and its validation metrics are presented below.

## Training Settings
- **Model Type**: YOLOv8s
- **Image Size**: 640
- **Dataset Fingerprint**: 72e11c0e37cf38f4
- **Gun Train Images**: 35,798
- **Hard Negatives**: 10,739

## Local Gradio Deployment
The newly trained model has been added to the local environment (`models/best.pt`) and a Gradio deployment script (`gradio_app.py`) has been created to test the model's visual performance manually. 
You can run the application locally with:
```bash
pip install gradio
python gradio_app.py
```

## Performance Metrics & Graphs
Below are the training graphs generated from the validation set during the detector runs.

### Results Curve
![Training Results](reports/images/results.png)

### Confusion Matrix
The confusion matrix shows the distribution of true positives versus false positives across different classes.
![Confusion Matrix](reports/images/confusion_matrix.png)

### Normalized Confusion Matrix
![Normalized Confusion Matrix](reports/images/confusion_matrix_normalized.png)

### Precision-Recall (PR) Curve
The Precision-Recall curve illustrates the trade-off between precision (fewer false positives) and recall (fewer false negatives). 
![Box PR Curve](reports/images/BoxPR_curve.png)

### Precision Curve
![Box Precision Curve](reports/images/BoxP_curve.png)

### Recall Curve
![Box Recall Curve](reports/images/BoxR_curve.png)

### F1 Curve
![Box F1 Curve](reports/images/BoxF1_curve.png)

## Conclusion
The model has been successfully extracted and implemented into the local deployment pipeline. Its real-world test performance can now be further assessed visually using the local Gradio interface.
