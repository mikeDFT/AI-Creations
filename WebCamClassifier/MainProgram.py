# includes
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
from ultralytics import YOLO  # Used for both pet and face detection

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

# for creating bounding boxes for pets
yolo_model = YOLO("yolov8n.pt")

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
	img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	face_tensor = mtcnn(img)
	if face_tensor is not None:
		if len(face_tensor.shape) == 3:
			face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension only if missing
		face_embedding = resnet(face_tensor.to(device))
		
		torch.save(face_embedding, face_embedding_path)
		print("✅ Owner face enrolled.")
	else:
		print("⚠️ No face detected. Try again.")


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
	
	if boxes is not None and len(boxes) > 0:
		for box in boxes:
			# print(box)
			x1, y1, x2, y2 = [int(i) for i in box]
			face_crop = frame[y1:y2, x1:x2]
			
			if face_crop.size == 0: # sometimes the bounding box is outside the frame
				continue
				
			face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
			cv2.imshow("Face", cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
				
			try:
				face_tensor = mtcnn(face_img)
				if face_tensor is not None:
					if len(face_tensor.shape) == 3:
						face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension only if missing
					face_embedding = resnet(face_tensor.to(device))
					
					similarity = torch.nn.functional.cosine_similarity(face_embedding, owner_embedding).item()
					if similarity > 0.8:
						label = "Owner"
					else:
						label = "Another Person"
					cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			except Exception as e:
				print(f"MTCNN failed: {e}")
			
	# --- Pet Detection ---
	yolo_results = yolo_model(frame, verbose=False)[0]  # Detect in the original frame
	for r in yolo_results.boxes.data.tolist():
		x1, y1, x2, y2, score, cls_id = r
		cls_id = int(cls_id)
		
		if cls_id in [15, 16] and score > 0.5:  # cat or dog
			detected_pet = True
			label = "Owner's Pet" if label == "Owner" else "Pet Only"
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
			cv2.putText(frame, "Pet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
	
	# Fallback label (no face or pet)
	if boxes is None and not detected_pet:
		label = "Nobody"
	
	cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
	cv2.imshow("AI Classifier", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

