import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "trained_model.h5"

CLASS_NAMES = [
    "iron_deficiency",
    "normal",
    "vitamin_a",
    "vitamin_b",
    "vitamin_c",
    "vitamin_d",
    "vitamin_e"
]

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return

    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)

    idx = np.argmax(pred)
    confidence = pred[0][idx]*100

    print("Prediction:",CLASS_NAMES[idx])
    print("Confidence:",round(confidence,2),"%")

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predict_image(image_path)