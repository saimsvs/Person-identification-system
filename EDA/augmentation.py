import cv2
import numpy as np
import os

# Input and output directories
input_dir = "/Users/saimasad/Desktop/Person-identification-system/balanced_images"    # Change this to your dataset path
output_dir = "augmented_dataset" # New folder where augmented images will be saved

# Create output folder if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through 11 person folders
for person_id in range(1, 12):
    folder_name = f"{person_id:02d}"
    person_folder = os.path.join(input_dir, folder_name)
    output_person_folder = os.path.join(output_dir, folder_name)

    if not os.path.exists(output_person_folder):
        os.makedirs(output_person_folder)

    images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png'))]

    for img_name in images:
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping: {img_path}")
            continue

        h, w = img.shape[:2]

        # --- Horizontal shift ---
        shift_pixels = int(0.1 * w)  # Shift by 10% width
        M = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
        shifted_img = cv2.warpAffine(img, M, (w, h))

        # --- Brightness adjustment ---
        bright_img = cv2.convertScaleAbs(img, alpha=1, beta=40)  # Increase brightness (beta)

        # --- Horizontal flip ---
        flipped_img = cv2.flip(img, 1)

        # Save original (if needed) and all augmentations
        base_name = os.path.splitext(img_name)[0]

        cv2.imwrite(os.path.join(output_person_folder, f"{base_name}_orig.jpg"), img)
        cv2.imwrite(os.path.join(output_person_folder, f"{base_name}_shift.jpg"), shifted_img)
        cv2.imwrite(os.path.join(output_person_folder, f"{base_name}_bright.jpg"), bright_img)
        cv2.imwrite(os.path.join(output_person_folder, f"{base_name}_flip.jpg"), flipped_img)

print("✅ Augmentation complete. Check 'augmented_dataset' folder.")