from ultralytics import YOLO

# Load YOLOv8 model (pretrained weights)
model = YOLO("yolov8n.pt")

# Train the model on your dataset
model.train(
    data="traffic_data.yaml",
    epochs=50,           # Increase if you want better accuracy
    imgsz=640,
    batch=8,
    device="cpu",        # use 'cuda' if you have GPU
    name="traffic_yolov8"
)
