import os
import random
import shutil

# Set your base (input) and output directories
base_dir = "/Users/saimasad/Desktop/Person-identification-system/cleaned_dataset"            # Each subfolder here = one person
output_dir = "balanced_images" # Output: balanced folders per person
os.makedirs(output_dir, exist_ok=True)

# Helper: get all image files (ignore hidden/non-image files)
def list_image_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')
    ])

# 1. Gather all folders and their image counts
person_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
counts = {}
image_files = {}
for person in person_folders:
    folder = os.path.join(base_dir, person)
    files = list_image_files(folder)
    image_files[person] = files
    counts[person] = len(files)

max_images = max(counts.values())
print(f"Max images per person: {max_images}")

# 2. Process each person
for person in person_folders:
    in_folder = os.path.join(base_dir, person)
    out_folder = os.path.join(output_dir, person)
    os.makedirs(out_folder, exist_ok=True)
    originals = image_files[person]

    # Copy all original images (skip if already copied)
    for fname in originals:
        src = os.path.join(in_folder, fname)
        dst = os.path.join(out_folder, fname)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Duplicate random images if needed
    current_files = list_image_files(out_folder)
    n_needed = max_images - len(current_files)
    copy_idx = 0
    while len(current_files) < max_images:
        fname = random.choice(originals)
        src = os.path.join(in_folder, fname)
        name, ext = os.path.splitext(fname)
        # Unique filename for the copy
        new_fname = f"{name}_copy{copy_idx}{ext}"
        dst = os.path.join(out_folder, new_fname)
        while os.path.exists(dst):
            copy_idx += 1
            new_fname = f"{name}_copy{copy_idx}{ext}"
            dst = os.path.join(out_folder, new_fname)
        shutil.copy(src, dst)
        current_files.append(new_fname)
        copy_idx += 1

    print(f"{person}: {len(list_image_files(out_folder))} images")

# Final check
print("\nFinal counts:")
for person in person_folders:
    out_folder = os.path.join(output_dir, person)
    total = len(list_image_files(out_folder))
    print(f"{person}: {total}")
    assert total == max_images, f"{person} has {total}, should be {max_images}"

print("All folders balanced successfully.")