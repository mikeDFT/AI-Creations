# includes
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
from ultralytics import YOLO  # used for both pet and face detection
from yolov5facedetector.detector import Yolov5FaceDetector


# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN for detecting faces (and making bounding boxes)
# keep_all makes it detect and return all faces it finds in the image,
# otherwise it'd return the largest one
mtcnn = MTCNN(keep_all=True, device=device)

# Inception ResNet V1 for face recognition (detect owner)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# pet classifier (also using resnet, but ResNet18, not Inception),
# for classifying between human and pet
weights = ResNet18_Weights.DEFAULT
pet_classifier = resnet18(weights=weights)
pet_classifier.eval()

# YOLO models for pet and face detection
yolo_pet_model = YOLO("yolov8n.pt") # for pets
yolo_face_model = YOLO("yolov5s-face.pt")  # for face detection

# --- Preprocessing ---
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
])

face_embedding_path = "owner_face.pt"

# --- Consent prompt ---
print("Webcam-based identity classifier (Owner | Pet | Stranger | Nobody)")
print("All processing is local. No data is transmitted.")
input("Press Enter to give consent and continue...")


# --- Enroll owner if not saved ---
def enroll_owner(frame):
	face_results = yolo_face_model(frame, verbose=False)[0]
	
	for r in face_results.boxes.data.tolist():
		x1, y1, x2, y2, score, cls_id = r
		if score < 0.6:
			continue
			
		face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
		if face_crop.size == 0:
			continue
			
		face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
		face_tensor = transform(face_img).unsqueeze(0).to(device)
		face_embedding = resnet(face_tensor)
		torch.save(face_embedding, face_embedding_path)
		
		print("✅ Owner face enrolled.")
		return
	
	print("⚠️ No valid face detected. Try again.")


if not os.path.exists(face_embedding_path):
	print("👤 Let's enroll your face as the owner.")
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		cv2.imshow("Enroll - Press 'e' to capture", frame)
		if cv2.waitKey(1) & 0xFF == ord('e'):
			enroll_owner(frame)
			break
	cap.release()
	cv2.destroyAllWindows()

owner_embedding = torch.load(face_embedding_path)

# --- Main loop ---
cap = cv2.VideoCapture(0)
print("📷 Starting webcam... Press 'q' to quit.")

while True:
	ret, frame = cap.read()
	if not ret:
		break
	
	img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	
	# Detect faces
	boxes, _ = mtcnn.detect(img)
	label = "Nobody"
	detected_pet = False
	
	# --- Face Detection ---
	face_results = yolo_face_model(frame, verbose=False)[0]
	for r in face_results.boxes.data.tolist():
		x1, y1, x2, y2, score, cls_id = r
		if score < 0.6:
			continue
		x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
		face_crop = frame[y1:y2, x1:x2]
		
		if face_crop.size == 0:
			continue
		
		face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
		face_tensor = transform(face_img).unsqueeze(0).to(device)
		face_embedding = resnet(face_tensor)
		similarity = torch.nn.functional.cosine_similarity(face_embedding, owner_embedding).item()
		
		label = "Owner" if similarity > 0.8 else "Another Person"
		
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			
	# --- Pet Detection ---
	pet_results = yolo_pet_model(frame, verbose=False)[0]
	for r in pet_results.boxes.data.tolist():
		x1, y1, x2, y2, score, cls_id = r
		cls_id = int(cls_id)
		if cls_id in [15, 16] and score > 0.5:  # cat or dog
			detected_pet = True
			label = "Owner's Pet" if label == "Owner" else "Pet Only"
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
			cv2.putText(frame, "Pet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
	
	# fallback label (no face or pet)
	if face_results.boxes.data.shape[0] == 0 and not detected_pet:
		label = "Nobody"
	
	cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
	cv2.imshow("AI Classifier", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

