from ultralytics import YOLO
import cv2
import os
import re

#Paths to your trained models
normal_model_path = r"C:\Users\harsh\OneDrive\Desktop\ATSC Project\models\normal_model\best.pt"
emergency_model_path = r"C:\Users\harsh\OneDrive\Desktop\ATSC Project\models\emergency_model\best2.pt"

#Load both models
normal_model = YOLO(normal_model_path)
emergency_model = YOLO(emergency_model_path)
print("✅ Both models loaded successfully!")

#Input test image path
image_path = r"C:\Users\harsh\OneDrive\Desktop\San_Francisco_(CA,_USA),_Powell_Street,_Polizeifahrzeuge_--_2022_--_2917.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found at {image_path}")

#Run inference
print("🚀 Running ensemble detection...")
results_normal = normal_model(image_path, conf=0.4, verbose=False)
results_emergency = emergency_model(image_path, conf=0.4, verbose=False)

#Read image
img = cv2.imread(image_path)

#Draw bounding boxes
def draw_boxes(result, color):
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{result.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Green → Normal vehicles, Red → Emergency vehicles
draw_boxes(results_normal[0], (0, 255, 0))
draw_boxes(results_emergency[0], (0, 0, 255))

#Create YOLO-style auto-incrementing folder
base_dir = r"runs\detect"
os.makedirs(base_dir, exist_ok=True)

# Find last folder index (predict, predict1, predict2...)
existing = [d for d in os.listdir(base_dir) if re.match(r"predict\d*$", d)]
if not existing:
    new_folder = "predict"
else:
    nums = [int(re.findall(r"\d+", d)[0]) for d in existing if re.findall(r"\d+", d)]
    next_num = max(nums) + 1 if nums else 1
    new_folder = f"predict{next_num}"

save_dir = os.path.join(base_dir, new_folder)
os.makedirs(save_dir, exist_ok=True)

#Save output image
output_path = os.path.join(save_dir, "ensemble_output.jpg")
cv2.imwrite(output_path, img)
print(f"\n✅ Ensemble detection complete! Saved output to:\n{output_path}")