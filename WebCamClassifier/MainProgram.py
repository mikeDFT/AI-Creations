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

# --- Consent prompt ---
print("Webcam-based identity classifier (Owner | Pet | Person | Nobody)")
print("All processing is local. No data is transmitted.")
input("Press Enter to give consent and continue...")

owner_embeddings = []
folder_path = 'owner_face_images'

# --- Enroll owner if not saved ---
def enroll_owner(frame, max_owner_face_img_nr):
	img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	faces = mtcnn(img)
	if faces is not None and len(faces) == 1:
		if len(faces.shape) == 3:
			face_tensor = faces.unsqueeze(0)  # Add batch dimension only if missing
		face_embedding = resnet(faces.to(device))
		
		max_owner_face_img_nr += 1
		torch.save(face_embedding, folder_path + "/" + str(max_owner_face_img_nr) + ".pt")
		
		print("✅ Owner face enrolled.")
	else:
		print("⚠️ No face or too many faces detected. Try again.")
		
	return max_owner_face_img_nr


def load_owner_face_pics():
	max_owner_face_img_nr = 0
	for filename in os.listdir(folder_path):
		face_embedding_path = os.path.join(folder_path, filename)
		
		# Check if the file ends with '.pt'
		if filename.endswith('.pt'):
			# Load the embedding from the .pt file
			embedding = torch.load(face_embedding_path)
			
			# Append the embedding to the list
			owner_embeddings.append(embedding)
			
			# the owner_face_images look like [nr].pt, so I'm updating that nr to the highest one
			# so as to be able to save more images
			max_owner_face_img_nr = max(int(filename[:-3]), max_owner_face_img_nr)
			
	return max_owner_face_img_nr


print("👤 Take pictures of your face as the owner")
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	cv2.imshow("Enroll - Press 'e' to capture, 'c' to continue", frame)
	
	max_owner_face_img_nr = load_owner_face_pics()
	
	waitKey = cv2.waitKey(1)
	if waitKey & 0xFF == ord('e'):
		max_owner_face_img_nr = enroll_owner(frame, max_owner_face_img_nr)
	if waitKey & 0xFF == ord('c'):
		break
		
cap.release()
cv2.destroyAllWindows()

# --- Main loop ---
cap = cv2.VideoCapture(0)
print("📷 Starting webcam... Press 'q' to quit.")

# function used to check if YOLO already detected a pet there
def IntersectionOverUnion(box1, box2):
	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])
	
	inter_area = max(0, x2 - x1) * max(0, y2 - y1)
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
	union_area = box1_area + box2_area - inter_area
	
	if union_area != 0:
		return inter_area / union_area
	else:
		return 0

def addToLabel(label, toAdd):
	if label == "Nobody":
		return toAdd
	else:
		return label + " & " + toAdd


while True:
	ret, frame = cap.read()
	if not ret:
		break
	
	img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	
	# Detect faces
	boxes, _ = mtcnn.detect(img)
	label = "Nobody"
	detected_pet = False
	detected_owner = False
	pet_boxes = []
	
	# --- Pet Detection ---
	yolo_results = yolo_model(frame, verbose=False)[0]  # Detect in the original frame
	for r in yolo_results.boxes.data.tolist():
		x1, y1, x2, y2, score, cls_id = r
		cls_id = int(cls_id)
		
		if cls_id in [15, 16] and score > 0.5:  # cat or dog
			detected_pet = True
			pet_boxes.append((x1, y1, x2, y2))
			
			label = addToLabel(label, "Pet")
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
			cv2.putText(frame, "Pet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
	
	if boxes is not None and len(boxes) > 0:
		for box in boxes:
			x1, y1, x2, y2 = [int(i) for i in box]
			face_crop = frame[y1:y2, x1:x2]
			
			if face_crop.size == 0: # sometimes the bounding box is outside the frame
				continue
			
			is_overlapping_pet = any(IntersectionOverUnion((x1, y1, x2, y2), pet_box) > 0.3 for pet_box in pet_boxes)
			if is_overlapping_pet:
				print("faces are overlapping")
				continue  # skip face box likely on a pet
				
			face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
			cv2.imshow("Face", cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
				
			try:
				face_tensor = mtcnn(face_img)
				if face_tensor is not None:
					if len(face_tensor.shape) == 3:
						face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension only if missing
					face_embedding = resnet(face_tensor.to(device))
					
					owner_embedding = torch.stack(owner_embeddings).mean(dim=0)
					
					similarity = torch.nn.functional.cosine_similarity(face_embedding, owner_embedding).item()
					if similarity > 0.7:
						boxLabel = "Owner"
						boxColor = (0, 150, 0)
					else:
						boxLabel = "Person"
						boxColor = (0, 255, 0)
						
					label = addToLabel(label, boxLabel)
					
					cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, 2)
					cv2.putText(frame, boxLabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, boxColor, 2)
			except Exception as e:
				print(f"MTCNN failed: {e}")
	
	cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
	cv2.imshow("AI Classifier", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

