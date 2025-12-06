import argparse
import os
import numpy as np
import tensorflow as tf

from ms_slices_loader import load_clean_dataset
from Models.UNet import build_unet
from Models.Attention_UNet import build_attention_unet
from Models.UNetPP import UNetPP


# -----------------------------
# Loss functions
# -----------------------------

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))


def weighted_bce_dice(y_true, y_pred):
    weight = 5.0
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return bce + weight * (1 - dice)


# -----------------------------
# Model selector
# -----------------------------

def get_model(model_name, input_shape):
    model_name = model_name.lower()

    if model_name == "unet":
        return build_unet(input_shape)

    elif model_name == "attention":
        return build_attention_unet(input_shape)

    elif model_name == "unetpp":
        return UNetPP(input_shape)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# -----------------------------
# Main training script
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Choose model: unet / attention / unetpp")
    args = parser.parse_args()

    print(f"\n🚀 Training model: {args.model}\n")

    # -----------------------------
    # Load dataset
    # -----------------------------

    train_dir = os.path.join("CleanedData", "train")
    test_dir = os.path.join("CleanedData", "test")

    X_train, Y_train = load_clean_dataset(train_dir)
    X_test, Y_test = load_clean_dataset(test_dir)

    if len(X_train) == 0:
        raise ValueError("❌ ERROR: No training samples loaded. Check CleanedData paths!")

    input_shape = X_train.shape[1:]

    # -----------------------------
    # Build model
    # -----------------------------
    model = get_model(args.model, input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce_dice,
        metrics=["accuracy"]
    )

    model.summary()

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=20,
        batch_size=8
    )

    # -----------------------------
    # Save trained model
    # -----------------------------
    os.makedirs("saved_models", exist_ok=True)
    model.save(f"saved_models/{args.model}.h5")

    print(f"\n🎉 Model saved → saved_models/{args.model}.h5\n")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
