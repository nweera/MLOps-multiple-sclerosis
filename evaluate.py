import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from Models.UNet import build_unet
from Models.Attention_UNet import build_attention_unet
from Models.UNetPP import UNetPP


# -------- Load PNG as Tensor ----------
def load_png(path):
    img = Image.open(path).convert("L")
    img = img.resize((256, 256))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    return arr


# ----------------- Metrics -----------------
def dice_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2 * intersection) / (y_true.sum() + y_pred.sum() + 1e-7)


def iou_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-7)


def precision_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    return tp / (tp + fp + 1e-7)


def recall_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    return tp / (tp + fn + 1e-7)


# -------- Make side-by-side image ----------
def make_triplet(original, gt, pred_bin, save_path):
    """
    original, gt, pred_bin are numpy arrays (256x256)
    """
    orig_img = Image.fromarray(original.astype("uint8"))
    gt_img = Image.fromarray(gt.astype("uint8"))
    pred_img = Image.fromarray(pred_bin.astype("uint8"))

    w, h = 256 * 3, 256 + 40
    canvas = Image.new("L", (w, h), color=255)

    canvas.paste(orig_img, (0, 40))
    canvas.paste(gt_img, (256, 40))
    canvas.paste(pred_img, (512, 40))

    draw = ImageDraw.Draw(canvas)
    draw.text((80, 10), "Original", fill=0)
    draw.text((320, 10), "Ground Truth", fill=0)
    draw.text((580, 10), "Predicted", fill=0)

    canvas.save(save_path)


# -------- Evaluate function ----------
def evaluate(model_name):
    print(f"\n🚀 Evaluating model: {model_name}")

    if model_name == "unet":
        model = build_unet((256, 256, 1))
        weight_path = "saved_models/unet.h5"

    elif model_name == "attention":
        model = build_attention_unet((256, 256, 1))
        weight_path = "saved_models/attention.h5"

    elif model_name == "unetpp":
        model = UNetPP((256, 256, 1))
        weight_path = "saved_models/unetpp.h5"

    else:
        raise ValueError("Unknown model. Use: unet / attention / unetpp")

    print(f"🔄 Loading weights from {weight_path}")
    model.load_weights(weight_path)
    print("✅ Model loaded successfully.\n")

    root = "CleanedData/test"
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")

    out_dir = os.path.join("evaluation_outputs", model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {img_dir} ...")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"📂 Found {len(files)} test examples\n")

    dice_scores, iou_scores, prec_scores, rec_scores = [], [], [], []

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            print(f"⚠ Missing mask for {fname}, skipping.")
            continue

        img_arr = load_png(img_path)[:, :, 0]
        mask_arr = load_png(mask_path)[:, :, 0] * 255  # GT in 0/255 format

        inp = np.expand_dims(np.expand_dims(img_arr, axis=-1), axis=0)
        pred = model.predict(inp, verbose=0)[0][:, :, 0]

        pred_bin = (pred > 0.5).astype("uint8") * 255

        # ----- METRICS -----
        dice_scores.append(dice_score(mask_arr, pred_bin))
        iou_scores.append(iou_score(mask_arr, pred_bin))
        prec_scores.append(precision_score(mask_arr, pred_bin))
        rec_scores.append(recall_score(mask_arr, pred_bin))

        # Save visualization
        save_path = os.path.join(out_dir, fname)
        make_triplet(img_arr * 255, mask_arr, pred_bin, save_path)

    # -------- Write results file --------
    results_path = os.path.join(out_dir, "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("====== Evaluation Results ======\n")
        f.write(f"Model: {model_name}\n\n")
        f.write(f"Average Dice:      {np.mean(dice_scores):.4f}\n")
        f.write(f"Average IoU:       {np.mean(iou_scores):.4f}\n")
        f.write(f"Average Precision: {np.mean(prec_scores):.4f}\n")
        f.write(f"Average Recall:    {np.mean(rec_scores):.4f}\n")

    print("\n🎉 Evaluation FINISHED!")
    print(f"📁 Results saved in: {out_dir}")
    print(f"📄 Metrics saved to: {results_path}\n")


# -------- MAIN --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="unet / attention / unetpp")
    args = parser.parse_args()

    evaluate(args.model)
