# Followed the tutorial here:
# https://www.tensorflow.org/tutorials/images/classification


import json
import os
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from keras import models, layers, utils

@dataclass
class Config:
    data_dir: str = "./images" # Path to the dataset
    out_dir: str = "./" # Path to the output directory
    batch_size: int = 32 # Number of images to process in each batch
    img_shape: Tuple[int, int, int] = (128, 128, 3) # Height, width, and rgb channels
    epochs: int = 10 # Number of times to iterate over the entire dataset
    data_split: float = 0.2 # Use 20% of the data for validation
    seed: int = 42 # Random seed for reproducibility


def create_sequential_model() -> models.Sequential:
    """
    Build a Sequential CNN for image datasets.
    Returns:
        A compiled Keras Sequential model.
    """
    model = models.Sequential([
        layers.Input(shape=Config.img_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')
    ])

    # https://victorzhou.com/blog/keras-cnn-tutorial/#4-compiling-the-model
    model.compile(
        optimizer="adam", # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    training, validation = utils.image_dataset_from_directory(
        directory=Config.data_dir,
        labels="inferred", # "inferred" means labels are generated from the directory structure
        image_size=Config.img_shape[:2],
        batch_size=Config.batch_size,
        label_mode="categorical", # Use categorical labels for the output
        validation_split=Config.data_split, # Split the data into training and validation sets
        subset="both", # Create both training and validation datasets
        seed=Config.seed,
        shuffle=True,
    )

    # [0, 255] range is not ideal for neural networks,
    # we should seek to make our input values small.
    # training.map(lambda x, y: (x / 255.0, y))
    # validation.map(lambda x, y: (x / 255.0, y))
    return training, validation


def train_and_export() -> None:
    """Train the model, export artifacts, and zip them under out_dir.zip."""
    os.makedirs(Config.out_dir, exist_ok=True)

    # Create training and validation data generators
    train_data, val_data = create_datasets()

    model = create_sequential_model()

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=Config.epochs, # Number of times to iterate
        verbose=1,
    )
    model.summary() # Print the model summary

    # Save final model and artifacts
    model_path = os.path.join(Config.out_dir, "model.keras")
    model.save(model_path)

    with open(os.path.join(Config.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(train_data.class_names, f, indent=2)

    print("Training complete. Artifacts:")
    print(f"- Model: {model_path}")
    print(f"- Labels: {os.path.join(Config.out_dir, 'labels.json')}")


def main() -> None:
    train_and_export()


if __name__ == "__main__":
    main()
