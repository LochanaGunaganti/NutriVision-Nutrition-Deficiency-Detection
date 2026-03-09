import os
import tensorflow as tf
import numpy as np
import cv2
import json

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "trained_model.h5")
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset", "combined")

IMG_SIZE = 224

# -----------------------------
# Load model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Load class names automatically
# -----------------------------
CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print("Loaded classes:", CLASS_NAMES)

# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Image not found")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    idx = np.argmax(preds)
    confidence = float(preds[0][idx])

    print("\n✅ Prediction:", CLASS_NAMES[idx])
    print("✅ Confidence:", round(confidence * 100, 2), "%")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    predict_image(image_path)
