#Load a clean dataset of images and masks from .nii files

import os
import cv2
import numpy as np

def load_clean_dataset(base_folder):
    images_dir = os.path.join(base_folder, "images")
    masks_dir  = os.path.join(base_folder, "masks")

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    X, Y = [], []

    for f in image_files:
        img_path = os.path.join(images_dir, f)
        mask_path = os.path.join(masks_dir, f)

        # mask must exist
        if not os.path.exists(mask_path):
            print(f"⚠️ Missing mask for {f}, skipping")
            continue

        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize (change to your model input size)
        img = cv2.resize(img, (128, 128)) / 255.0
        mask = cv2.resize(mask, (128, 128)) / 255.0

        # Expand channels
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        Y.append(mask)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"Loaded {len(X)} samples from {base_folder}")
    return X, Y
