# Person Identification System

A Python-based system for identifying individuals from facial images, featuring data balancing, feature extraction, model training, and a graphical user interface.

---

## Features

- Face detection using MediaPipe.
- Feature extraction with HOG (Histogram of Oriented Gradients) and LBP (Local Binary Patterns).
- Data balancing and augmentation scripts.
- Decision Tree-based model training and evaluation.
- Tkinter GUI for image and webcam-based identification.
- Real-time identification via webcam.

---

## Repository Structure

```text
main.py
featureextraction_hog.py
featureextractionwlbp.py
featureextractionwoutlbp.py
aug_features.csv
final_features.csv
decision_tree_hog_model_combined.pkl
```

---

## Setup & Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/saimsvs/Person-identification-system.git
    cd Person-identification-system
    ```

2. **Install dependencies**
    ```bash
    pip install opencv-python mediapipe scikit-learn scikit-image pandas pillow joblib
    ```
    *(Or use a requirements file if available.)*

3. **Prepare your dataset**
    - Place cleaned face images for each person in a `cleaned_dataset/` subdirectory (not included in the repo).

---

## Usage

### 1. Feature Extraction

Extract HOG or LBP features:
```bash
python featureextraction_hog.py
# or
python featureextractionwlbp.py
```

### 2. Model Training

Train the Decision Tree classifier:
```bash
python final_training.py
```
- The trained model is saved as `decision_tree_hog_model_combined.pkl`.

### 3. Run the GUI Application

Launch the person identification system:
```bash
python main.py
```

---

## How it Works

- Detects faces, extracts features, normalizes them, and predicts identity using a Decision Tree model.
- Supports static images and live webcam streams.
- GUI displays prediction and bounding box for recognized faces.

---

## Authors

- Saim Asad
- Hasnain Abbas
- Junaid Ahmed

---

## License

No license specified. Please contact the repository owner for usage permissions.

---

## Acknowledgments

- Built using OpenCV, scikit-learn, MediaPipe, scikit-image, Pillow, Pandas, Tkinter, Joblib.
