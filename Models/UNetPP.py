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
    x = Conv2D(filters, (3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# -----------------------------
# UNet++ Architecture
# -----------------------------
def UNetPP(input_shape=(128, 128, 1)):

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

    return Model(inputs=[inputs], outputs=[outputs])


# -----------------------------
# Weighted BCE + Dice Loss
# -----------------------------
def weighted_bce_dice(y_true, y_pred):
    weight = 5.0

    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)

    return bce + weight * (1 - dice)


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":

    model = UNetPP()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce_dice
    )

    model.summary()
