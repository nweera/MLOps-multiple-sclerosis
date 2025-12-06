#!/usr/bin/env python3
"""
Attention U-Net implementation (clean, parameterized, and ready-to-use).

Usage:
    from attention_unet import build_attention_unet, compile_model
    model = build_attention_unet(input_shape=(128,128,1), base_filters=32, dropout=0.0)
    compile_model(model, lr=1e-4, loss="binary_crossentropy", metrics=["accuracy"])
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Activation,
    BatchNormalization,
    Multiply,
    Add,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------
# Attention Gate
# ---------------------------------------------------------
def attention_block(x, g, inter_channels, name=None):
    """Applies an attention gate to skip connections."""
    theta_x = Conv2D(inter_channels, 1, padding="same",
                     name=f"{name}_theta" if name else None)(x)
    phi_g = Conv2D(inter_channels, 1, padding="same",
                   name=f"{name}_phi" if name else None)(g)

    add_xg = Add(name=f"{name}_add" if name else None)([theta_x, phi_g])
    relu_xg = Activation("relu", name=f"{name}_relu" if name else None)(add_xg)

    psi = Conv2D(1, 1, padding="same", name=f"{name}_psi" if name else None)(relu_xg)
    sigmoid_psi = Activation("sigmoid",
                             name=f"{name}_sigmoid" if name else None)(psi)

    return Multiply(name=f"{name}_multiply" if name else None)([x, sigmoid_psi])


# ---------------------------------------------------------
# Convolution block
# ---------------------------------------------------------
def conv_block(x, filters, kernel_size=3, use_batchnorm=True, dropout=0.0, name=None):
    conv_name = (lambda s: None if not name else f"{name}_{s}")

    x = Conv2D(filters, kernel_size, padding="same", name=conv_name("conv1"))(x)
    if use_batchnorm:
        x = BatchNormalization(name=conv_name("bn1"))(x)
    x = Activation("relu", name=conv_name("act1"))(x)

    if dropout > 0.0:
        x = Dropout(dropout, name=conv_name("dropout"))(x)

    x = Conv2D(filters, kernel_size, padding="same", name=conv_name("conv2"))(x)
    if use_batchnorm:
        x = BatchNormalization(name=conv_name("bn2"))(x)
    x = Activation("relu", name=conv_name("act2"))(x)

    return x


# ---------------------------------------------------------
# Build Attention UNet
# ---------------------------------------------------------
def build_attention_unet(
    input_shape=(128, 128, 1),
    base_filters=32,
    dropout=0.0,
    use_batchnorm=True,
    num_classes=1,
    final_activation="sigmoid",
):
    inputs = Input(shape=input_shape, name="input")

    # Encoder
    c1 = conv_block(inputs, base_filters, use_batchnorm=use_batchnorm, dropout=dropout, name="enc1")
    p1 = MaxPooling2D(2, name="pool1")(c1)

    c2 = conv_block(p1, base_filters * 2, use_batchnorm=use_batchnorm, dropout=dropout, name="enc2")
    p2 = MaxPooling2D(2, name="pool2")(c2)

    c3 = conv_block(p2, base_filters * 4, use_batchnorm=use_batchnorm, dropout=dropout, name="enc3")
    p3 = MaxPooling2D(2, name="pool3")(c3)

    c4 = conv_block(p3, base_filters * 8, use_batchnorm=use_batchnorm, dropout=dropout, name="enc4")
    p4 = MaxPooling2D(2, name="pool4")(c4)

    # Bridge
    c5 = conv_block(p4, base_filters * 16, use_batchnorm=use_batchnorm, dropout=dropout, name="bridge")

    # Decoder + Attention
    u6 = UpSampling2D(2, name="up6")(c5)
    a6 = attention_block(c4, u6, base_filters * 8, name="att6")
    c6 = conv_block(Concatenate(name="concat6")([u6, a6]),
                    base_filters * 8, use_batchnorm=use_batchnorm, dropout=dropout, name="dec6")

    u7 = UpSampling2D(2, name="up7")(c6)
    a7 = attention_block(c3, u7, base_filters * 4, name="att7")
    c7 = conv_block(Concatenate(name="concat7")([u7, a7]),
                    base_filters * 4, use_batchnorm=use_batchnorm, dropout=dropout, name="dec7")

    u8 = UpSampling2D(2, name="up8")(c7)
    a8 = attention_block(c2, u8, base_filters * 2, name="att8")
    c8 = conv_block(Concatenate(name="concat8")([u8, a8]),
                    base_filters * 2, use_batchnorm=use_batchnorm, dropout=dropout, name="dec8")

    u9 = UpSampling2D(2, name="up9")(c8)
    a9 = attention_block(c1, u9, base_filters, name="att9")
    c9 = conv_block(Concatenate(name="concat9")([u9, a9]),
                    base_filters, use_batchnorm=use_batchnorm, dropout=dropout, name="dec9")

    # Output
    if num_classes == 1:
        activation = final_activation or "sigmoid"
        outputs = Conv2D(1, 1, padding="same", activation=activation, name="output")(c9)
    else:
        activation = final_activation or "softmax"
        outputs = Conv2D(num_classes, 1, padding="same", activation=activation, name="output")(c9)

    return Model(inputs, outputs, name="Attention_UNet")


# ---------------------------------------------------------
# Compile function
# ---------------------------------------------------------
def compile_model(model, lr=1e-4, loss="binary_crossentropy", metrics=None):
    if metrics is None:
        metrics = ["accuracy"]

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )


# ---------------------------------------------------------
# Demo (optional)
# ---------------------------------------------------------
if __name__ == "__main__":
    model = build_attention_unet(input_shape=(128, 128, 1))
    compile_model(model)
    model.summary()
