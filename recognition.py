import os
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
from PIL import Image
from ultralytics import YOLO
import cv2, pickle, faiss, torch
import numpy as np
from collections import Counter
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
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
 
index = faiss.read_index("object_index.faiss")
with open("object_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
 
match_count = Counter()
 
cap = cv2.VideoCapture(0)
recognized_names=[]
while True:
    ret, frame = cap.read()
    results = yolo_model(frame)
 
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
 
            features = extract_features(cropped).astype(np.float32).reshape(1, -1)
            D, I = index.search(features, k=1)
 
            if D[0][0] > 0.5:
                matched_id=I[0][0]
                name=metadata[matched_id].get("name", f"Object_{matched_id}")
                label = f"matched"
                recognized_names.append(name)
            else:
                label = "Unknown"
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.imshow("Recognize Objects", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()


print("\n Object Recognition Summary:")
for name in set(recognized_names):
    print(f"-{name} ({recognized_names.count(name)} times)")
# for idx, count in match_count.items():
#     print(f"Object #{idx}: Seen {count} times")

command = [
    "ollama", "run", "llama3.2",
    f'give summary, "{recognized_names}"'
]

try:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8',errors='replace')
 
    print("llama3.2 streaming output:\n")
    for line in iter(process.stdout.readline, ''):
        print(line.strip())  # Or stream this to UI / WebSocket etc.
 
    process.stdout.close()
    process.wait()
 
except subprocess.CalledProcessError as e:
    print("Error running llama3.2:", e.stderr)

# try:
#     result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
#     print("llama3.2 output:\n", result.stdout)
# except subprocess.CalledProcessError as e:
#     print("Error running llava:", e.stderr)
