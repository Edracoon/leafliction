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
import rembg
import os
import sys
import numpy as np
from keras.models import load_model, Sequential
from keras.preprocessing import image
import matplotlib.pyplot as plt


MODEL_PATH = 'model/model.keras'
LABELS_PATH = 'model/labels.json'


def preprocess_image(image_path: str):
    img = image.load_img(image_path, target_size=(128, 128))

    original_img = img.copy()

    img = rembg.remove(img, bgcolor=(255, 255, 255))
    img = img.convert('RGB')

    # Convert the image to a numpy array
    img = image.img_to_array(img)

    # Add a dimension for the batch
    img = np.expand_dims(img, axis=0)
    return original_img, img


def predict_label(model: Sequential, classes: list[str], img: np.ndarray):
    """
    Predict the label and confidence of the image.
    """
    probs = model.predict(img, verbose=0)[0]

    # Get the class with the highest probability
    predicted_class = int(np.argmax(probs))

    # Get label and confidence
    label = classes[predicted_class]
    conf = float(probs[predicted_class])

    return label, conf


def display_results(original_img, used_img: np.ndarray, prediction: str):
    _, axes = plt.subplots(1, 2, figsize=(15, 5 * 2))

    axes[0].imshow(original_img, interpolation='nearest')
    axes[0].set_title('Original 128x128', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    used_show = np.clip(used_img, 0, 255).astype(np.uint8)
    axes[1].imshow(used_show, interpolation='nearest')
    axes[1].set_title('Used 128x128', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    # Full screen and show
    plt.title(prediction, fontsize=12, fontweight='bold')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.tight_layout()
    plt.show()
    plt.savefig("outputs/predict.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args(argv: list[str]):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict disease from a leaf image."
    )
    parser.add_argument("image", help="Path to input image")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(2)

    img, img_data = preprocess_image(args.image)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    model: Sequential = load_model(MODEL_PATH)
    label, conf = predict_label(model, class_indices, img_data)
    prediction = f"Class predicted : {label}  (conf. {conf:.2%})"

    print(prediction)
    display_results(img, img_data[0], prediction)

    print()


if __name__ == "__main__":
    main()
