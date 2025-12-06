import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Activation, BatchNormalization, concatenate
)
from tensorflow.keras.models import Model


# -----------------------------
# Convolutional Block
# -----------------------------
def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# -----------------------------
# U-Net Architecture
# -----------------------------
def build_unet(input_shape=(256, 256, 1)):

    inputs = Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, 512)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 32)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs, outputs)
    return model


# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":

    model = build_unet((256, 256, 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=lambda y_true, y_pred:
            0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) +
            0.5 * dice_loss(y_true, y_pred),
        metrics=["accuracy"]
    )

    model.summary()
