import socket
import numpy as np
import cv2
from PIL import Image
import faiss
import os, pickle
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet50_Weights
from ultralytics import YOLO

# -------- Load YOLO & ResNet Feature Extractor --------
yolo_model = YOLO("yolov8n.pt")
weights = ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights).eval()
transform = weights.transforms()
extractor = create_feature_extractor(resnet, return_nodes={"avgpool": "features"})

# -------- Load MiDaS Depth Estimator --------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# -------- FAISS Setup --------
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

# -------- Normalize Helper --------
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

# -------- Feature Extraction --------
def extract_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    with torch.no_grad():
        input_tensor = transform(pil_img).unsqueeze(0)
        output = extractor(input_tensor)["features"]
        features = output.squeeze().numpy().reshape(-1)
        return normalize(features)

# -------- Depth Estimation --------
def estimate_depth(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    input_tensor = midas_transforms(pil_image).unsqueeze(0)
    with torch.no_grad():
        prediction = midas(input_tensor)
        depth = prediction.squeeze().cpu().numpy()
        return depth

# -------- UDP Receiver --------
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"[INFO] Listening for UDP video stream on port {UDP_PORT}...")

# -------- Frame Receive Loop --------
try:
    while True:
        packet, _ = sock.recvfrom(65536)
        np_data = np.frombuffer(packet, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2]
                features = extract_features(cropped).astype(np.float32).reshape(1, -1)

                index.add(features)
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]

                # Estimate depth and position
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                depth_map = estimate_depth(frame)
                bbox_depth = depth_map[y1:y2, x1:x2]
                distance_estimate = float(np.median(bbox_depth))

                metadata.append({
                    'image': cropped,
                    'name': class_name,
                    'position': (x_center, y_center),
                    'distance': distance_estimate
                })

                print(f"Object: {class_name}, Pos: {(x_center, y_center)}, Distance: {distance_estimate:.2f}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO + Depth + FAISS", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Shutting down...")

finally:
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    sock.close()
    cv2.destroyAllWindows()

# print(f"added objects: {class_name}")
