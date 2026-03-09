import os
import cv2

RAW = "dataset/raw_images"
COMBINED = "dataset/combined"

IMG_SIZE = 224

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(COMBINED, exist_ok=True)

classes = os.listdir(RAW)

for cls in classes:

    src_folder = os.path.join(RAW, cls)
    dst_folder = os.path.join(COMBINED, cls)

    os.makedirs(dst_folder, exist_ok=True)

    for file in os.listdir(src_folder):

        path = os.path.join(src_folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If face detected → crop
        if len(faces) > 0:

            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]

        else:
            # If no face detected → use original image
            face = img

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        save_path = os.path.join(dst_folder, file)

        cv2.imwrite(save_path, face)

print("Preprocessing complete")