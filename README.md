# Person Identification System

A Python-based system for identifying individuals from facial images, featuring data balancing, feature extraction, model training, and a graphical user interface.

---

## Repository Structure

```
EDA/
  ├── datasetcreator.py
  ├── augment.py
  └── balance.py
featureextraction/
  ├── featureextraction_hog.py
  ├── featureextractionwlbp.py
  └── featureextractionwoutlbp.py
ML_training/
  ├── training.py
  └── final_training.py
main.py
decision_tree_hog_model_combined.pkl
```
##Features
- Feature detection using MediaPipe.
- Feature extraction with HOG(Histogram of Oriented Gradients) and LBP(Local Binary Patterns).
- Balancing Data and Augmentation.
- Decision tree base model training.
- Tkinter GUI for image and webcam-based identification.
- Real-time identification through webcam.
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
    - Place face images for each person in a `dataset/` subdirectory (not included in the repo).
      

---

## Usage

### 1. Data Preparation

Balance and augment images (if using balancing/augmentation scripts):
 This would be first ran through the file datasetcreator.py next balance.py and finally augment.py.
```bash
python eda/balance.py
python eda/augment.py
python eda/datasetcreator.py
```
### 2. Feature Extraction

Extract HOG or LBP features:
In the model we have features through both techniques but trained it on one technique and received features are then used for ML.
```bash
python featureextraction/featureextraction_hog.py
# or
python featureextraction/featureextractionwlbp.py
```
### 3. Model Training

Train the Decision Tree classifier:
```bash
python ML_training/final_training.py
```
- The trained model is saved as `decision_tree_hog_model_combined.pkl`.

### 4. Run the GUI Application

Launch the person identification system:
```bash
python main.py
```

---

## How it Works
- Detects the face ,extracts features and normalises them to predict identity of an individual.
- Providess an option for static images and a live webcam detection.
- The GUI would show live prediction.
---
## Authors

- Saim Asad
- Hasnain Abbas
- Junaid Ahmed

---

## Dependencies & Libraries Used

- Built using OpenCV, scikit-learn, MediaPipe, scikit-image, Pillow, Pandas, Tkinter, Joblib.

