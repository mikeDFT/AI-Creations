# includes
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
from ultralytics import YOLO  # Used for both pet and face detection
import Cryptography

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

# for segmentation
seg_model = YOLO("yolov8n-seg.pt")

owner_embeddings = [] # A face embedding is a list of numbers (a tensor) that uniquely represents a face.
folder_path = 'owner_face_images'

colors = []
for i in range(20):
	colors.append(tuple(np.random.randint(0, 256, size=3).tolist()))

# --- Consent prompt ---
def consentPrompt():
	print("Webcam-based identity classifier (Owner | Pet | Person | Nobody)")
	print("All processing is local, no API or any external calls. No data is transmitted.")
	print(
		"All of the owner's facial images are encrypted, saved locally and only processed locally by pretrained models")
	answer = input(
		"Type 'opt out' if you want all your data to be erased or press Enter to give consent and continue\n> ")
	if answer == "opt out":
		for filename in os.listdir(folder_path):
			face_embedding_path = os.path.join(folder_path, filename)
			# Check if the file ends with '.pt'
			if filename.endswith('.pt'):
				os.remove(face_embedding_path)
		
		Cryptography.remove_key()
		
		print("Successfully removed all locally saved data")
		exit()

# --- Enroll owner ---
def enroll_owner(frame, max_owner_face_img_nr):
	img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	faces = mtcnn(img)
	if faces is not None and len(faces) == 1:
		if len(faces.shape) == 3:
			faces = faces.unsqueeze(0)  # Add dimension only if missing
			
		face_embedding = resnet(faces.to(device))
		owner_embeddings.append(face_embedding)
		
		max_owner_face_img_nr += 1
		Cryptography.encrypt_embedding(face_embedding, os.path.join(folder_path, f"{max_owner_face_img_nr}.pt"))
		
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
			embedding = Cryptography.decrypt_embedding(face_embedding_path)
			
			# Append the embedding to the list
			owner_embeddings.append(embedding)
			
			# the owner_face_images look like [nr].pt, so I'm updating that nr to the highest one
			# so as to be able to save more images
			max_owner_face_img_nr = max(int(filename[:-3]), max_owner_face_img_nr)
			
	return max_owner_face_img_nr


def takePicturesLoop():
	print("👤 Take pictures of your face as the owner")
	print("Press 'e' to capture, 'c' to continue, 'q' to quit")
	cap = cv2.VideoCapture(0)
	
	max_owner_face_img_nr = load_owner_face_pics()
	bounding_boxes = []
	frame_count = -1
	facesLabel = "None"
	
	while True:
		ret, frame = cap.read()
		frame_count += 1
		
		# change to %2 to render every 2 frames or to %1 to not use this
		if frame_count % 1 == 0:
			facesLabel = "None"
			img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			
			bounding_boxes, newFacesLabel = processMTCNNFaces(img, frame, facesLabel)
			if newFacesLabel:
				facesLabel = newFacesLabel
		
		for boxTuple in bounding_boxes:
			applyBoundingBox(frame, boxTuple)
		
		cv2.putText(frame, "Faces: " + facesLabel, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
		cv2.imshow("Enroll - Press 'e' to capture, 'c' to continue, 'q' to quit", frame)
		
		waitKey = cv2.waitKey(1)
		if waitKey & 0xFF == ord('e'):
			max_owner_face_img_nr = enroll_owner(frame, max_owner_face_img_nr)
		if waitKey & 0xFF == ord('c'):
			if len(owner_embeddings) == 0:
				print("No owner embeddings (pictures) provided, please capture pictures by pressing 'e'")
			else:
				break
		if waitKey & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			exit()
	
	cap.release()
	cv2.destroyAllWindows()


def processYOLOObjectDetection(frame, show_only_restricted_classes, restrictedClasses, pet_boxes, label):
	# --- Object Detection ---
	yolo_results = yolo_model(frame, verbose=False)[0]  # Detect in the original frame
	bounding_boxes = []
	
	for data_result in yolo_results.boxes.data.tolist():
		x1, y1, x2, y2, score, cls_id = data_result
		cls_id = int(cls_id)
		
		if score > 0.5:
			class_name = yolo_model.names[cls_id]  # Get the human-readable class name
			if show_only_restricted_classes and class_name not in restrictedClasses:
				continue
			
			pet_boxes.append((x1, y1, x2, y2))
			
			color = ((20 * cls_id + 170) % 213, (60 * cls_id + 60) % 213, (100 * cls_id + 10) % 213)
			# cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
			# cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
			
			bounding_boxes.append((
				((int(x1), int(y1)), (int(x2), int(y2)), color, 2),
				(class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8)
			))
			
			label = addToLabel(label, class_name)
			
	return bounding_boxes, label

def processMTCNNFaces(img, frame, facesLabel):
	# process the faces from MTCNN
	boxes, _ = mtcnn.detect(img)
	bounding_boxes = []
	
	if boxes is not None and len(boxes) > 0:
		for box in boxes:
			x1, y1, x2, y2 = [int(i) for i in box]
			face_crop = frame[y1:y2, x1:x2]
			
			if face_crop.size == 0:  # sometimes the bounding box is outside the frame
				continue
			
			# is_overlapping_pet = any(IntersectionOverUnion((x1, y1, x2, y2), pet_box) > 0.3 for pet_box in pet_boxes)
			# if is_overlapping_pet:
			# 	print("faces are overlapping")
			# 	continue  # skip face box likely on a pet
			
			face_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
			# showing a window with the face
			# cv2.imshow("Face", cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
			
			try:
				face_tensor = mtcnn(face_img)
				if face_tensor is not None:
					if len(face_tensor.shape) == 3:
						face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension only if missing
					face_embedding = resnet(face_tensor.to(device))
					
					# Stacking: combining multiple embedding vectors into a single tensor to process or compare them together.
					# basically makes it a list of embeddings
					# then I take their mean, which makes it a general representation of my face
					owner_embedding = torch.stack(owner_embeddings).mean(dim=0)
					
					similarity = torch.nn.functional.cosine_similarity(face_embedding, owner_embedding).item()
					if similarity > 0.7:
						boxLabel = "Owner"
						boxColor = (0, 150, 0)
					else:
						boxLabel = "Stranger"
						boxColor = (0, 255, 0)
					
					# cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, 2)
					# cv2.putText(frame, boxLabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, boxColor, 2)
					
					bounding_boxes.append((
						((int(x1), int(y1)), (int(x2), int(y2)), boxColor, 2),
						(boxLabel, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8)
					))
					
					facesLabel = addToLabel(facesLabel, boxLabel)
			except Exception as e:
				print(f"MTCNN failed: {e}")
	
	return bounding_boxes, facesLabel

def applyBoundingBox(frame, boundingBoxTuple):
	pt1, pt2, color, thickness = boundingBoxTuple[0]
	text, org, font, fontScale = boundingBoxTuple[1]
	
	cv2.rectangle(frame, pt1, pt2, color, thickness)
	cv2.putText(frame, text, org, font, fontScale, color, thickness)
	# cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
	# cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def processSegmentation(frame, show_only_restricted_classes, restrictedClasses):
	seg_results = seg_model(frame, verbose=False)[0]
	seg_masks = seg_results.masks
	
	class_ids = seg_results.boxes.cls.cpu().numpy()
	
	masks_list = []
	
	if seg_masks is not None:
		mask_nr = 0
		for (mask, cls_id) in zip(seg_masks.data, class_ids):
			if show_only_restricted_classes:
				class_name = yolo_model.names[cls_id]  # Get the human-readable class name
				if show_only_restricted_classes and class_name not in restrictedClasses:
					continue
			
			mask = mask.cpu().numpy()
			binary_mask = mask > 0.5
			
			mask_layer = np.zeros_like(frame, dtype=np.uint8)
			mask_layer[binary_mask] = colors[mask_nr]
			
			alpha = 0.5
			masks_list.append((binary_mask, mask_layer, alpha))
			mask_nr += 1
			
	return masks_list

def applyMask(frame, maskTuple):
	binary_mask, mask_layer, alpha = maskTuple
	frame[binary_mask] = cv2.addWeighted(frame[binary_mask], 1 - alpha, mask_layer[binary_mask], alpha, 0)

# --- Main loop ---
# function used to check if YOLO already detected a pet there
# if the intersection is quite large, compared to the union, then the boxes are on top of one another
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

# I use this code in many places, so I made a function
def addToLabel(givenLabel, toAdd):
	if givenLabel == "Nobody" or givenLabel == "None":
		return toAdd
	else:
		return givenLabel + ", " + toAdd

def main():
	cap = cv2.VideoCapture(0)
	
	show_segmentation = False
	show_only_restricted_classes = True
	restrictedClasses = ["dog", "cat", "person"]
	
	masks_list = []
	YOLO_bounding_boxes = []
	MTCNN_bounding_boxes = []
	frame_count = -1
	label = ""
	facesLabel = ""
	
	print("📷 Starting webcam... Press 'q' to quit, 's' to turn on/off segmentation, 'r' to turn on/off restricted classes")
	
	while True:
		frame_count += 1
		ret, frame = cap.read()
		if not ret:
			break
		
		# change to %2 to render every 2 frames or to %1 to not use this
		if frame_count % 1 == 0:
			img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			
			label = "Nobody"
			facesLabel = "None"
			pet_boxes = []
			
			YOLO_bounding_boxes, label = processYOLOObjectDetection(frame, show_only_restricted_classes, restrictedClasses, pet_boxes, label)
			
			MTCNN_bounding_boxes, facesLabel = processMTCNNFaces(img, frame, facesLabel)
			
			masks_list = []
			if show_segmentation:
				masks_list = processSegmentation(frame, show_only_restricted_classes, restrictedClasses)
			
			
		# applying the bounding boxes and masks so that I can also apply them from other frames
		# to be able to skip frames and make it faster
		for boxTuple in YOLO_bounding_boxes:
			applyBoundingBox(frame, boxTuple)
		
		for boxTuple in MTCNN_bounding_boxes:
			applyBoundingBox(frame, boxTuple)
		
		if show_segmentation:
			for maskTuple in masks_list:
				applyMask(frame, maskTuple)
		
		cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
		cv2.putText(frame, "Faces: " + (facesLabel or "None"), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
		
		cv2.imshow("AI Classifier - Press 'q' to quit, 's' to turn on/off segmentation, 'r' to turn on/off restricted classes", frame)
		
		waitKey = cv2.waitKey(1)
		if waitKey & 0xFF == ord('q'):
			break
		if waitKey & 0xFF == ord('s'):
			show_segmentation = not show_segmentation
			print(f"Segmentation {'enabled' if show_segmentation else 'disabled'}")
		if waitKey & 0xFF == ord('r'):
			show_only_restricted_classes = not show_only_restricted_classes
			print(f"Only restricted classes {'enabled' if show_only_restricted_classes else 'disabled'}")

	cap.release()
	cv2.destroyAllWindows()


consentPrompt()
takePicturesLoop()
main()
