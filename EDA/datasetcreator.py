import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from PIL import Image
from tqdm import tqdm



# Constants
DATASET_DIR = "dataset"
CLEANED_DIR = "cleaned_dataset"
IMAGE_SIZE = (256, 256)

# Create cleaned dataset directory
os.makedirs(CLEANED_DIR, exist_ok=True)

# Initialize MediaPipe and the face detector so we can crop images from the background
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


rows = []
for folder in sorted([f for f in os.listdir(DATASET_DIR) if f.isdigit()], key=lambda x: int(x)):
# loop over the data set and alos convert the strings to int
    person_folder = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(person_folder):# skip the nondirectory and joi the clean directory with each person
        continue

    label = int(folder) - 1  
    cleaned_person_folder = os.path.join(CLEANED_DIR, folder)
    os.makedirs(cleaned_person_folder, exist_ok=True)

    for image_name in tqdm(sorted(os.listdir(person_folder)), desc=f"Processing Person {folder}"):
        image_path = os.path.join(person_folder, image_name)
        cleaned_image_path = os.path.join(cleaned_person_folder, image_name)
  
  #read image sand cob=vert to rgb for media pipe as it needs rgb
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)
#the cropping,covertoing to gray scale, adn resizing is done here
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x_min = max(int(bbox.xmin * w), 0)
                    y_min = max(int(bbox.ymin * h), 0)
                    x_max = min(int((bbox.xmin + bbox.width) * w), w)
                    y_max = min(int((bbox.ymin + bbox.height) * h), h)

                    face_img = img[y_min:y_max, x_min:x_max]
                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, IMAGE_SIZE)
                    cv2.imwrite(cleaned_image_path, face_resized)

                    rows.append({"image_path": cleaned_image_path, "label": label})
                   
                    break  
                # Take only one face per image
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


eda_df = pd.DataFrame(rows)
print(eda_df.head())

