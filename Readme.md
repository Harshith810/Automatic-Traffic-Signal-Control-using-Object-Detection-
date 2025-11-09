Automatic Traffic signal Control System (ATSC) with Emergency Vehicle Detection

This an AI-powered solution that uses object detection (YOLOv8) to manage urban traffic more efficiently and sustainably. The system analyzes real-time traffic footage to determine vehicle density and detect emergency vehicles such as ambulances and fire trucks. Based on these insights, it dynamically adjusts traffic light durations, ensuring smoother flow, quicker emergency response, and reduced emissions.


Features

Real-Time Vehicle Detection: Identifies and counts cars, bikes, buses, and trucks.

Adaptive Signal Control: Adjusts green light duration based on real-time traffic density.

Emergency Vehicle Priority: Detects emergency vehicles and prioritizes their lane.

Eco-Friendly Design: Minimizes idle time and reduces CO₂ emissions.


Tech Stack

Programming Language :Python 3

AI Model : YOLOv8 (Ultralytics)

Computer Vision : OpenCV

Data Annotation : LabelImg / Roboflow Annotate

System Workflow
1. Input: Real-time or recorded traffic video.
2. Detection: YOLOv8 identifies vehicle types and detects emergency vehicles.
3. Analysis: Counts vehicles in each lane to determine density.
4. Signal Control: Adjusts traffic light duration or provides emergency clearance.
5. Output: Updated traffic light sequence optimizing flow and reducing congestion.

Week 2 Progress:

week 2 progress focused on training, fine-tuning, and testing models for both normal and emergency traffic conditions:

1  Model Development and Training

Collected and organized datasets for both normal vehicles and emergency vehicles.

Preprocessed and structured data with separate folders for train and val sets.

Trained two independent YOLOv8 models:

Normal Traffic Model: Detects general vehicle types (car, bus, truck, bike).

Emergency Vehicle Model: Detects emergency classes (ambulance, fire truck, police car).

Trained Normal Traffic Model on VS code (atsc_train.py).

Trained Emergency Vehicle Model on Google Colab using GPU acceleration (emergency_train.ipynb).

Saved final trained models (best.pt) to Google Drive.

2 Fine-Tuning and Optimization

Fine-tuned the normal model using the emergency dataset to improve recognition of rare classes.

Evaluated models using precision, recall, mAP50, and mAP50-95 metrics. 

And achieved 90.7 mAP50(accuracy) for emergency vehicle detection model and 88.3 mAP50(accuracy) for normal vehicle detection model 

Compared performances between fine-tuned and standalone models, due to the dataset imbalance the fine-tuned model is not detecting accurately so figured out its better to use ensembling strategy with two individual models

Implemented YOLO model ensembling strategy for improved combined detection accuracy.

3 Testing and Validation

Tested both models individually and in ensemble form using custom test images.

Verified detection results visually using bounding boxes for each detected class.

Identified an issue where the emergency model confused ambulance with police cars and started retraining with a refined dataset (emergency.zip).

Again tested the ensemble form and its is working fine and the detected output image will be automatically saved into the device (runs/detect/pretict/ensemble_output).

4 Model Management

Automated saving of trained models and weights (best.pt, last.pt) to Google Drive.

Exported trained models for use in VS Code environment for deployment testing.

