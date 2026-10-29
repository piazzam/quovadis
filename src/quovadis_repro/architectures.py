from __future__ import annotations

from typing import Any


def build_keras_model(name: str, num_classes: int) -> Any:
    """Build the exact from-scratch architectures reported in the paper."""
    from tensorflow.keras import Input, layers, models

    normalized = name.lower()
    if normalized in {"cnn_s", "mnist_cnn", "mnist"}:
        inputs = Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs=inputs, outputs=outputs, name="cnn_s")

    if normalized in {"cnn_d_fashion", "fashion_mnist_cnn", "fashion_mnist"}:
        input_shape = (28, 28, 1)
    elif normalized in {"cnn_d_cifar10", "cifar10_cnn", "cifar10"}:
        input_shape = (32, 32, 3)
    else:
        raise ValueError(f"Unknown Keras architecture: {name}")

    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name=normalized)
