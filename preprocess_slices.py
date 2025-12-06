import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_dir = r"Data"

# Collect all (pid, image_slice, mask_slice)
pairs = []

for patient_folder in tqdm(os.listdir(data_dir)):
    patient_path = os.path.join(data_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    pid = patient_folder.split("-")[-1]

    flair_file = os.path.join(patient_path, f"{pid}-Flair.nii")
    mask_file = os.path.join(patient_path, f"{pid}-LesionSeg-Flair.nii")

    if not (os.path.exists(flair_file) and os.path.exists(mask_file)):
        print(f"Skipping {patient_folder}: Missing files")
        continue

    flair = nib.load(flair_file).get_fdata()
    mask = nib.load(mask_file).get_fdata()

    flair = np.asarray(flair)
    mask = np.asarray(mask)

    mid = flair.shape[2] // 2
    slice_indices = [mid - 1, mid, mid + 1]

    for idx in slice_indices:
        if idx < 0 or idx >= flair.shape[2]:
            continue  # safety check

        flair_slice = np.rot90(flair[:, :, idx])
        mask_slice = np.rot90(mask[:, :, idx])

        flair_slice = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min()) * 255
        flair_slice = flair_slice.astype(np.uint8)

        mask_slice = (mask_slice > 0).astype(np.uint8) * 255

        # store slice number in filename
        pairs.append((pid, idx - mid, flair_slice, mask_slice))

print("Extracted slices. Splitting into train/test...")

train_set, test_set = train_test_split(pairs, test_size=0.2, random_state=42)

def save_data(split_name, data):
    img_dir = os.path.join("CleanedData", split_name, "images")
    mask_dir = os.path.join("CleanedData", split_name, "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for pid, relative_idx, img, msk in data:
        # relative_idx = -1, 0, or +1
        slice_name = f"middle{relative_idx:+d}"
        filename = f"Patient-{pid}_{slice_name}.png"

        plt.imsave(os.path.join(img_dir, filename), img, cmap="gray")
        plt.imsave(os.path.join(mask_dir, filename), msk, cmap="gray")

save_data("train", train_set)
save_data("test", test_set)

print("Done! Saved 3 slices per patient.")
