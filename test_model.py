#!/usr/bin/env python3
"""
test_model.py — Tests all images in test_images/ against the trained model.
Displays expected label (from filename) vs predicted label with confidence.

Usage:
    python3 test_model.py
"""

import os
import re
import json
from keras.models import load_model
from d_data_classification.Predict import (
    preprocess_image, predict_label, MODEL_PATH, LABELS_PATH
)


TEST_DIR = 'test_images'


def extract_expected_label(filename: str) -> str:
    """Extract expected label from filename.
    e.g. 'Apple_Black_rot1.JPG' -> 'Apple_Black_rot'
    """
    name = os.path.splitext(filename)[0]
    return re.sub(r'\d+$', '', name)


def main():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = load_model(MODEL_PATH)

    total = 0
    correct = 0

    for subdir in sorted(os.listdir(TEST_DIR)):
        subdir_path = os.path.join(TEST_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue

        print(f"\n{'=' * 60}")
        print(f"  {subdir}")
        print(f"{'=' * 60}")

        for filename in sorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, filename)
            if not os.path.isfile(filepath):
                continue

            expected = extract_expected_label(filename)
            _, img_data = preprocess_image(filepath)
            predicted, conf = predict_label(model, classes, img_data)

            match = expected == predicted
            status = "OK" if match else "KO"
            total += 1
            correct += int(match)

            print(
                f"  [{status}] {filename:<30}"
                f"  expected: {expected:<20}"
                f"  predicted: {predicted:<20}"
                f"  (conf. {conf:.2%})"
            )

    print(f"\n{'=' * 60}")
    print(f"  Results: {correct}/{total} correct")
    if total > 0:
        print(f"  Accuracy: {correct / total:.2%}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
