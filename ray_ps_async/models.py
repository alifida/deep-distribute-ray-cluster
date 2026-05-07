from typing import Tuple

import tensorflow as tf


def build_model(model_name: str, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    if model_name != "tiny":
        try:
            base = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
            )
            base.trainable = False
            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
            out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            return tf.keras.Model(inputs=base.input, outputs=out)
        except Exception:
            pass

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)

