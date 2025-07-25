import os
import cv2
import numpy as np
import joblib
import os
import cv2
import numpy as np
import joblib
import pandas as pd
from skimage.feature import hog
import mediapipe as mp
from tkinter import Tk, Button, Label, Frame, filedialog, Canvas
from PIL import Image, ImageTk

# Constants
IMAGE_SIZE = (128, 128)
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_BLOCK_NORM = 'L2-Hys'
DIST_THRESHOLD = 200  # Tune as needed

# Load model and normalization params

model_data = joblib.load("decision_tree_hog_model_combined.pkl")
model = model_data["model"]
mean = model_data["mean"]
std = model_data["std"]

# Load training HOG features from CSV (for webcam only)

df = pd.read_csv("/Users/saimasad/Desktop/Person-identification-system/aug_features.csv")
X_train = df.iloc[:, 2:].values  # Skip 'image_path' and 'label' columns
X_train_scaled = (X_train - mean) / (std + 1e-10)


# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Webcam state
cap = None
is_webcam_running = False

def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if results.detections:
        d = results.detections[0]
        bbox = d.location_data.relative_bounding_box
        h, w, _ = img.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            return None, img, None
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)
        return resized, img, (x1, y1, x2, y2)
    return None, img, None

def extract_features(img):
    features = hog(img, 
                   orientations=HOG_ORIENTATIONS, 
                   pixels_per_cell=HOG_PIXELS_PER_CELL, 
                   cells_per_block=HOG_CELLS_PER_BLOCK, 
                   block_norm=HOG_BLOCK_NORM)
    return (features - mean) / (std + 1e-10)

def predict_person(img, use_aug_features=False):
    if model is None:
        return None, img, None
    face_img, original, bbox = preprocess_image(img)
    if face_img is not None:
        features = extract_features(face_img)
        if use_aug_features and X_train_scaled is not None:  # For webcam
            distances = np.linalg.norm(X_train_scaled - features, axis=1)
            min_dist = np.min(distances)
            if min_dist > DIST_THRESHOLD:
                return "unknown", original, bbox
            else:
                pred = model.predict([features])[0]
                return pred, original, bbox
        else:  # For images
            pred = model.predict([features])[0]
            return pred, original, bbox
    return None, original, None

def display_result():
    if model is None:
        result_label.config(text="Error: Model not loaded")
        return
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if path:
        img = cv2.imread(path)
        if img is None:
            result_label.config(text="Error loading image")
            return
        pred, img, bbox = predict_person(img, use_aug_features=False)  # No aug_features for images
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((400, 400))
        tk_img = ImageTk.PhotoImage(pil_img)
        image_label.config(image=tk_img, width=400, height=400)
        image_label.image = tk_img
        image_label.pack()
        if pred is not None:
            result_label.config(text=f"Predicted ID: {pred + 1}")
        else:
            result_label.config(text="Face not detected")

def update_webcam():
    global cap, is_webcam_running
    if not is_webcam_running:
        return
    ret, frame = cap.read()
    if ret:
        pred, img, bbox = predict_person(frame, use_aug_features=True)  # Use aug_features for webcam
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((400, 400))
        tk_img = ImageTk.PhotoImage(pil_img)
        image_label.config(image=tk_img, width=400, height=400)
        image_label.image = tk_img
        image_label.pack()
        if pred == "unknown":
            result_label.config(text="Person not recognized (unknown)")
        elif pred is not None:
            result_label.config(text=f"Predicted ID: {pred + 1}")
        else:
            result_label.config(text="Face not detected")
        root.after(60, update_webcam)  # ~16.7 FPS
    else:
        stop_webcam()

def start_webcam():
    global cap, is_webcam_running
    if model is None:
        result_label.config(text="Error: Model not loaded")
        return
    if X_train_scaled is None:
        result_label.config(text="Error: Training features not loaded")
        return
    if not is_webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            result_label.config(text="Error: Could not open webcam")
            return
        is_webcam_running = True
        select_btn.config(state="disabled")
        webcam_btn.config(text="Close Webcam", command=stop_webcam)
        update_webcam()

def stop_webcam():
    global cap, is_webcam_running
    if is_webcam_running:
        cap.release()
        is_webcam_running = False
        image_label.config(image="", width=0, height=0)
        image_label.pack_forget()
        result_label.config(text="")
        select_btn.config(state="normal")
        webcam_btn.config(text="Open Webcam", command=start_webcam)

def show_welcome_screen():
    splash = Tk()
    splash.overrideredirect(True)
    splash.geometry("600x300+400+200")
    splash.attributes("-alpha", 0.92)
    splash.configure(bg="#2B2B3C")
    title = Label(splash, text="Welcome to our Person Identification Model", font=("Segoe UI", 20, "bold"), fg="#E1BEE7", bg="#2B2B3C")
    title.pack(pady=40)
    subtitle = Label(splash, text="by Saim Asad, Hasnain Abbas, Junaid Ahmed", font=("Segoe UI", 18), fg="#C5CAE9", bg="#2B2B3C")
    subtitle.pack(pady=10)
    splash.after(3000, splash.destroy)
    splash.mainloop()

# GUI setup
show_welcome_screen()
root = Tk()
root.title("Person Identifier")
root.configure(bg="#1E1E2F")
root.geometry("540x640")
header = Label(root, text="Person Identifier", font=("Segoe UI", 22, "bold"), fg="#E0E0E0", bg="#1E1E2F")
header.pack(pady=20)
btn_frame = Frame(root, bg="#1E1E2F")
btn_frame.pack(pady=10)
select_btn = Button(
    btn_frame,
    text="Select Image",
    command=display_result,
    font=("Segoe UI", 12, "bold"),
    bg="#6A1B9A",
    fg="white",
    activebackground="#8E24AA",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=12,
    bd=0,
    highlightthickness=0,
)
select_btn.pack(side="left", padx=10)
webcam_btn = Button(
    btn_frame,
    text="Open Webcam",
    command=start_webcam,
    font=("Segoe UI", 12, "bold"),
    bg="#6A1B9A",
    fg="white",
    activebackground="#8E24AA",
    activeforeground="white",
    relief="flat",
    padx=20,
    pady=12,
    bd=0,
    highlightthickness=0,
)
webcam_btn.pack(side="left", padx=10)
image_border = Frame(root, bg="#1E1E2F")
image_border.pack(pady=20)
image_label = Label(image_border)
image_label.pack_forget()
result_canvas = Canvas(root, width=500, height=40, bg="#2C2C3A", highlightthickness=0)
result_canvas.pack(pady=10)
result_label = Label(root, text="", font=("Segoe UI", 14), fg="#E0E0E0", bg="#1E1E2F")
result_label.pack()
root.mainloop()