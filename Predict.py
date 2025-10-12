#!/usr/bin/env python3
"""
Predict.py — Charge ./model.zip (contenant model.keras et labels.json),
affiche/sauvegarde une vue Original vs Transformed (sans matplotlib) et
imprime la prédiction.

Usage simple:
    python3 Predict.py ./path_to_img.jpg

Options:
    --img-size H W       Taille d'entrée du modèle (défaut 224 224)
    --show               Ouvre la visualisation avec le viewer système (PIL)
    --save-vis PATH      Sauvegarde l'image côte-à-côte (JPG/PNG)
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from keras.models import load_model, Sequential
from keras.preprocessing import image


MODEL_PATH = 'model/model.keras'
LABELS_PATH = 'model/labels.json'

def preprocess_image(image_path: str) -> np.ndarray:
    img = image.load_img(image_path, target_size=(128, 128))
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Add a dimension for the batch
    img = np.expand_dims(img, axis=0)
    return img


def predict_label(model: Sequential, class_indices: [str], img: np.ndarray) -> Tuple[str, float]:
    """
    Predict the label and confidence of the image.
    """
    probs = model.predict(img, verbose=0)[0]

    # Get the class with the highest probability
    predicted_class = int(np.argmax(probs))

    # Get label and confidence
    label = class_indices[predicted_class]
    conf = float(probs[predicted_class])

    return label, conf


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Predict disease from a leaf image.")
    parser.add_argument("image", help="Path to input image")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(2)

    img = preprocess_image(args.image)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    model: Sequential = load_model(MODEL_PATH)
    label, conf = predict_label(model, class_indices, img)

    print(f"Prediction: {label}  (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
