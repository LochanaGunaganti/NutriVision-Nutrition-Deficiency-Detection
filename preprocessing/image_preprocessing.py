import os
import shutil

RAW = "dataset/raw_images"
COMBINED = "dataset/combined"

IMAGE_EXT = (".jpg", ".jpeg", ".png")

os.makedirs(COMBINED, exist_ok=True)

for class_name in os.listdir(RAW):

    src_folder = os.path.join(RAW, class_name)
    dst_folder = os.path.join(COMBINED, class_name)

    os.makedirs(dst_folder, exist_ok=True)

    for file in os.listdir(src_folder):

        src_file = os.path.join(src_folder, file)

        if os.path.isfile(src_file) and file.lower().endswith(IMAGE_EXT):

            shutil.copy(src_file, os.path.join(dst_folder, file))

print("Images copied successfully to dataset/combined")