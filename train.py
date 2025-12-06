import argparse
import os
import yaml
import tensorflow as tf

from ms_slices_loader import load_clean_dataset
from Models.UNet import build_unet


# -----------------------------
# Loss functions
# -----------------------------

def weighted_bce_dice(y_true, y_pred):
    weight = 5.0
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return bce + weight * (1 - dice)


# -----------------------------
# UNet model loader
# -----------------------------

def get_model(input_shape):
    print("📌 Using UNet (fixed model).")
    return build_unet(input_shape)


# -----------------------------
# Main training script
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    # -----------------------------
    # Load YAML config
    # -----------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_dir = config["paths"]["train_dir"]
    test_dir = config["paths"]["test_dir"]
    save_dir = config["paths"]["save_dir"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]

    print("\n🚀 Loaded config:")
    print(config)

    # -----------------------------
    # Load dataset
    # -----------------------------

    X_train, Y_train = load_clean_dataset(train_dir)
    X_test, Y_test = load_clean_dataset(test_dir)

    if len(X_train) == 0:
        raise ValueError("❌ ERROR: No training samples loaded. Check CleanedData paths!")

    input_shape = X_train.shape[1:]

    # -----------------------------
    # Build UNet
    # -----------------------------
    model = get_model(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=weighted_bce_dice,
        metrics=["accuracy"]
    )

    model.summary()

    # -----------------------------
    # Train the model
    # -----------------------------
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    # -----------------------------
    # Save trained model
    # -----------------------------
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "unet.h5")
    model.save(model_path)

    print(f"\n🎉 Model saved → {model_path}\n")


if __name__ == "__main__":
    main()
