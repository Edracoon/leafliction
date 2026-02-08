# Followed the tutorial here:
# https://www.tensorflow.org/tutorials/images/classification


import json
import os
from pathlib import Path
import random
import shutil
from dataclasses import dataclass
from typing import Tuple
from keras import models, layers, utils


@dataclass
class Config:
    data_dir: str = "./augmented_directory"  # Path to the dataset
    out_dir: str = "./model"  # Path to the output directory
    batch_size: int = 32  # Number of images to process in each batch
    img_shape: Tuple[int, int, int] = (128, 128, 3)  # Height, width, and rgb
    epochs: int = 5  # Number of times to iterate over the entire dataset
    split_validation: float = 0.15  # Use 15% of the data for validation
    split_test: float = 0.15  # Use 15% of the data for testing
    seed: int = 42  # Random seed for reproducibility


def create_sequential_model():
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
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets():
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    training = utils.image_dataset_from_directory(
        directory=Config.data_dir + "/train",
        image_size=Config.img_shape[:2],
        batch_size=Config.batch_size,
        label_mode="categorical",  # Use categorical labels for the output
        seed=Config.seed,
    )

    validation = utils.image_dataset_from_directory(
        directory=Config.data_dir + "/validation",
        image_size=Config.img_shape[:2],
        batch_size=Config.batch_size,
        label_mode="categorical",  # Use categorical labels for the output
        seed=Config.seed,
    )

    return (training, validation)


def train_and_export():
    """Train the model, export artifacts, and zip them under out_dir.zip."""
    os.makedirs(Config.out_dir, exist_ok=True)

    # Create training and validation data generators
    train_data, val_data = load_datasets()

    model = create_sequential_model()

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=Config.epochs,  # Number of times to iterate
        verbose=1,
    )
    model.summary()  # Print the model summary

    # Save final model and artifacts
    model_path = os.path.join(Config.out_dir, "model.keras")
    model.save(model_path)

    with open(os.path.join(Config.out_dir, "labels.json"), "w") as f:
        json.dump(train_data.class_names, f, indent=2)

    print("Training complete. Artifacts:")
    print(f"- Model: {model_path}")
    print(f"- Labels: {os.path.join(Config.out_dir, 'labels.json')}")


def split_dataset(dir: str):
    """
    Split dataset training / validation / test sets to ensure
    no overfitting (70% training, 15% validation, 15% tests).
    """
    # Create destination folders
    root = Path(dir)

    train_dir = root / "train"
    val_dir = root / "validation"
    test_dir = root / "test"

    if train_dir.exists() or val_dir.exists() or test_dir.exists():
        print("Warning: train/validation/test directories already exist.")
        return

    subdir_classes = [d for d in root.iterdir() if d.is_dir()]

    for subdir in subdir_classes:
        imgs = [p for p in subdir.iterdir() if p.is_file()]
        if not imgs:
            print(f"Warning: no images in {subdir}")
            continue

        random.shuffle(imgs)

        total = len(imgs)
        val = int(total * Config.split_validation)
        test = int(total * Config.split_test)

        train_imgs = imgs[:total - val - test]  # 1 -> 70 %
        val_imgs = imgs[total - val - test:total - test]  # 71 -> 85 %
        test_imgs = imgs[total - test:]  # 86 -> 100 %

        (train_dir / subdir.name).mkdir(parents=True, exist_ok=True)
        for f in train_imgs:
            shutil.copy2(str(f), str(train_dir / subdir.name / f.name))

        (val_dir / subdir.name).mkdir(parents=True, exist_ok=True)
        for f in val_imgs:
            shutil.copy2(str(f), str(val_dir / subdir.name / f.name))

        (test_dir / subdir.name).mkdir(parents=True, exist_ok=True)
        for f in test_imgs:
            shutil.copy2(str(f), str(test_dir / subdir.name / f.name))


def main():
    split_dataset(Config.data_dir)
    train_and_export()


if __name__ == "__main__":
    main()
