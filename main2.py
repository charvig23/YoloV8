from ultralytics import YOLO
import cv2, pickle, os
from PIL import Image
import faiss
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from datetime import datetime

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load ResNet50 for feature extraction
weights = ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights).eval()
transform = weights.transforms()
extractor = create_feature_extractor(resnet, return_nodes={"avgpool": "features"})

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def extract_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    with torch.no_grad():
        input_tensor = transform(pil_img).unsqueeze(0)
        output = extractor(input_tensor)["features"]
        features = output.squeeze().numpy().reshape(-1)
        return normalize(features)

# Initialize FAISS and metadata
feature_dim = 2048
index_path = "object_index.faiss"
metadata_path = "object_metadata.pkl"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatIP(feature_dim)

if os.path.exists(metadata_path):
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
else:
    metadata = []

# Start video stream (IP Webcam or MJPEG source)
cap = cv2.VideoCapture("http://[2405:201:4003:4085:8b7d:19a5:bb52:44d6]:8080/video")

if not cap.isOpened():
    raise RuntimeError("❌ Could not open video stream. Check the IP or port.")

frame_count = 0
save_crops = True
os.makedirs("detections", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]

            # Extract features
            features = extract_features(cropped).astype(np.float32).reshape(1, -1)
            index.add(features)

            # Get object class
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Save metadata
            metadata.append({'name': class_name, 'timestamp': timestamp})

            print(f"[{timestamp}] Detected: {class_name} at ({x1},{y1})-({x2},{y2})")

            # Optional: save cropped image
            if save_crops:
                crop_path = f"detections/{timestamp}_frame{frame_count}_{class_name}.jpg"
                cv2.imwrite(crop_path, cropped)

    frame_count += 1

    # Optional: break loop after N frames (to avoid infinite loop in test)
    # if frame_count > 100:
    #     break

# Save index and metadata
faiss.write_index(index, index_path)
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

cap.release()

print(f"✅ Processed {frame_count} frames. Added {len(metadata)} objects.")
