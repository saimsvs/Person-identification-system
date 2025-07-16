import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog

balanced_dir = "balanced_images"
IMAGE_SIZE = (256, 256)  # Already resized in your cleaned dataset
FEATURE_CSV = "final_features.csv"

# the same model will be used to produce the csv file for augmented dataset. you would have to 
# change the balanced dir and destination.

# HOG Parameters
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_BLOCK_NORM = 'L2-Hys'

# lists to store data
features = []
labels = []
paths = []

# Loop through the cleaned dataset
for folder in sorted(os.listdir(balanced_dir), key=lambda x: int(x)):
    person_folder = os.path.join(balanced_dir, folder)
    if not os.path.isdir(person_folder):
        continue

    label = int(folder) - 1  # 1-indexed to 0-indexed

    for image_name in tqdm(sorted(os.listdir(person_folder)), desc=f"Extracting HOG for Person {folder}"):
        image_path = os.path.join(person_folder, image_name)

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # HOG feature extraction
            hog_features = hog(img,
                               orientations=HOG_ORIENTATIONS,
                               pixels_per_cell=HOG_PIXELS_PER_CELL,
                               cells_per_block=HOG_CELLS_PER_BLOCK,
                               block_norm=HOG_BLOCK_NORM,
                               visualize=False,
                               feature_vector=True)

            features.append(hog_features)
            labels.append(label)
            paths.append(image_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# convert to numpy arrays
X = np.array(features)
y = np.array(labels)
paths = np.array(paths)

# create and save DataFrame
df = pd.DataFrame(X)
df.insert(0, "label", y)
df.insert(0, "image_path", paths)
df.to_csv(FEATURE_CSV, index=False)

print(f"HOG feature vectors saved to {FEATURE_CSV}")
print(df.head())