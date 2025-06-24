from ultralytics import YOLO
import cv2, pickle, os
from PIL import Image
import faiss
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
 

yolo_model = YOLO("yolov8n.pt")
 
 
 
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

cap = cv2.VideoCapture("http://192.168.29.7:8080/video")
ret, frame = cap.read()
if not ret or frame is None:
    print("Failed to capture video frame. Check IP address and connection.")
    exit()

while True:
    ret, frame = cap.read()
    results = yolo_model(frame)
 
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            features = extract_features(cropped).astype(np.float32).reshape(1, -1)
 
            print("Feature shape:", features.shape)
            print("Expected FAISS dim:", index.d)
 
            index.add(features)
            class_id=int(box.cls[0])
            class_name=yolo_model.names[class_id]
            metadata.append({'image': cropped, 'name':class_name})  
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Added', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
 
    cv2.imshow("Registering Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

faiss.write_index(index, index_path)
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
 
cap.release()
cv2.destroyAllWindows()
print(f"added objects: {class_name}")
