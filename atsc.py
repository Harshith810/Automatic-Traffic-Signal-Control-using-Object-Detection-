
from ultralytics import YOLO


import cv2              # For video processing
import numpy as np      # For mathematical operations
import matplotlib.pyplot as plt  
from PIL import Image            # For image handling

# Load model
model = YOLO('yolov8n.pt')
results = model.predict(source="test_image/traffic_img.jpg", save=True)