#!/usr/bin/env python3
"""
Batch augmentation program for processing multiple images.
Automatically applies data augmentation to the first N images in a directory.
"""

import os
import sys
import random
import glob
import rembg
from PIL import Image
from tqdm import tqdm

from b_data_augmentation.Augmentation import augment_image
from a_data_analysis.Distribution import compute_distribution


def get_image_files(directory_path: str):
    """
    Gets the first N image files from the specified directory.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return []

    # Get all JPG files in the directory
    image_pattern = os.path.join(directory_path, "*.JPG")
    image_files = glob.glob(image_pattern)

    # Sort files to ensure consistent order
    image_files.sort()

    # Return only the first N images
    return image_files


def remove_all_backgrounds(directory_path: str):
    """
    Remove backgrounds from all images in all subdirectories.
    """
    images = []
    for root, _, files in os.walk(directory_path):
        for f in files:
            if os.path.splitext(f)[1].lower().endswith('.jpg'):
                images.append(os.path.join(root, f))

    session = rembg.new_session("u2netp")

    for img_path in tqdm(images, desc="Removing backgrounds"):
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((512, 512))
        img = rembg.remove(img, session=session, bgcolor=(255, 255, 255))
        img = img.convert('RGB')
        img.save(img_path)


def augment_balance_directory(directory_path: str):
    """
    Runs augmentations on the subdirectories to balance the dataset.
    """
    # Remove backgrounds from all images first
    remove_all_backgrounds(directory_path)

    # Compute the distribution of images in the directory
    # { "dir_name": count, ... }
    distribution = compute_distribution(directory_path)

    biggest_dir_name = max(distribution, key=lambda key: distribution[key])
    biggest_dir_count = distribution[biggest_dir_name]

    # Delete the biggest directory to avoid processing it for augmentation
    distribution.pop(biggest_dir_name)

    total = 0
    for key in distribution:
        augment_number = int((biggest_dir_count - distribution[key]) / 6)
        images = os.listdir(os.path.join(sys.argv[1], key))
        print(f"\n\n-- Augmenting {key} with {augment_number} images --")
        for i in range(augment_number):
            print(f'\r> {i+1}/{augment_number} images', end='')
            cur_img = random.choice(images)
            augment_image(os.path.join(sys.argv[1], key, cur_img), False)
            images.remove(cur_img)
            total += 1

    print("\n" + "=" * 60)
    print(f"Batch augmentation completed for {total} images!")
    print("All images have been saved in their respective directories.")


def main():
    """
    Main function to handle command line arguments and execute batch.
    """
    if len(sys.argv) != 2:
        print("Usage: python3 BatchAugmentation.py <directory_path>")
        print("Example: python3 BatchAugmentation.py ./images")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Run batch augmentation
    augment_balance_directory(directory_path)


if __name__ == "__main__":
    main()
