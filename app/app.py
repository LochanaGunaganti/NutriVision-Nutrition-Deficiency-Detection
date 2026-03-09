import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "trained_model.h5")

IMG_SIZE = 224

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# =====================================================
# CLASS LABELS (MUST MATCH TRAINING ORDER)
# =====================================================

CLASS_NAMES = [
    "iron_deficiency",
    "normal",
    "vitamin_a",
    "vitamin_b",
    "vitamin_c",
    "vitamin_d",
    "vitamin_e"
]

print("Model classes:", CLASS_NAMES)

# =====================================================
# FOOD RECOMMENDATIONS
# =====================================================

FOOD_SUGGESTIONS = {

    "vitamin_a": [
        {"name": "Carrot", "image": "carrot.jpg"},
        {"name": "Spinach", "image": "spinach.jpg"},
        {"name": "Sweet Potato", "image": "sweet_potato.jpg"}
    ],

    "vitamin_b": [
        {"name": "Egg", "image": "egg.jpg"},
        {"name": "Milk", "image": "milk.jpg"}
    ],

    "vitamin_c": [
        {"name": "Orange", "image": "orange.jpg"},
        {"name": "Lemon", "image": "lemon.jpg"},
        {"name": "Amla", "image": "amla.jpg"}
    ],

    "vitamin_d": [
        {"name": "Fish", "image": "fish.jpg"},
        {"name": "Egg", "image": "egg.jpg"},
        {"name": "Milk", "image": "milk.jpg"}
    ],

    "vitamin_e": [
        {"name": "Almonds", "image": "almonds.jpg"},
        {"name": "Sunflower Seeds", "image": "sunflower.jpg"}
    ],

    "iron_deficiency": [
        {"name": "Spinach", "image": "spinach.jpg"},
        {"name": "Beans", "image": "beans.jpg"},
        {"name": "Red Meat", "image": "meat.jpg"}
    ],

    "normal": []
}

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(file):

    try:

        file.seek(0)

        file_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return None

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img / 255.0

        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:

        print("Image preprocessing error:", e)

        return None


# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")


# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        files = request.files.getlist('images')

        print("Images received:", len(files))

        if len(files) == 0:
            return jsonify({"error": "No images uploaded"}), 400

        predictions = []

        # --------------------------------
        # Run prediction on each image
        # --------------------------------

        for file in files:

            img = preprocess_image(file)

            if img is None:
                continue

            pred = model.predict(img, verbose=0)

            predictions.append(pred[0])

        if len(predictions) == 0:
            return jsonify({"error": "Invalid images"}), 400

        predictions = np.array(predictions)

        # --------------------------------
        # Average predictions
        # --------------------------------

        final_prediction = np.mean(predictions, axis=0)

        # --------------------------------
        # Top 3 deficiencies
        # --------------------------------

        top_indices = final_prediction.argsort()[-3:][::-1]

        top_results = []

        for idx in top_indices:

            top_results.append({
    "deficiency": str(CLASS_NAMES[idx]),
    "confidence": float(round(float(final_prediction[idx])*100,2))
})

        # --------------------------------
        # Main deficiency
        # --------------------------------

        main_idx = top_indices[0]

        predicted_class = CLASS_NAMES[main_idx]

        confidence = round(float(final_prediction[main_idx]) * 100, 2)

        # --------------------------------
        # Health Score
        # --------------------------------

        avg_deficiency = np.mean(final_prediction) * 100

        health_score = round(100 - avg_deficiency, 2)

        if health_score > 70:
            health_status = "Good"
        elif health_score > 40:
            health_status = "Moderate"
        else:
            health_status = "Poor"

        # --------------------------------
        # Risk Level
        # --------------------------------

        if confidence < 30:
            risk = "Low"
        elif confidence < 60:
            risk = "Moderate"
        else:
            risk = "High"

        foods = FOOD_SUGGESTIONS.get(predicted_class, [])

        # --------------------------------
        # Return result
        # --------------------------------

        result = {
    "main_deficiency": str(predicted_class),
    "confidence": float(confidence),
    "risk_level": str(risk),
    "health_score": float(health_score),
    "health_status": str(health_status),
    "top_deficiencies": top_results,
    "foods": foods,
    "images_used": int(len(files))
}
        print("Prediction result:", result)

        return jsonify(result)

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({"error": str(e)}), 500


@app.route('/result')
def result():
    return render_template("result.html")


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)