"""
Data augmentation program for balancing image datasets.
6 types of augmentations: Flip, Rotate, Skew, Shear, Crop, Distortion.
"""

import os
import sys
import cv2
import numpy as np
import rembg
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import random


def remove_background(image: np.ndarray) -> np.ndarray:
    """Remove background from an OpenCV image using rembg."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img = rembg.remove(pil_img, bgcolor=(255, 255, 255))
    pil_img = pil_img.convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_image(image_path: str):
    """
    Loads an image from the given path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image '{image_path}'.")
    return image


def apply_flip(image: np.ndarray):
    """
    Applies vertical flip to the image.
    """
    return cv2.flip(image, 1)


def apply_rotate(image: np.ndarray):
    """
    Applies random rotation to the image.
    """
    # Random rotation angle between -90 and 90 degrees
    angle = random.uniform(-90, 90)

    # Get image dimensions
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated


def apply_skew(image: np.ndarray):
    """
    Applies skew transformation to the image.
    """
    # It consists of a rotation, and a shear.
    rotated = apply_rotate(image)
    skewed = apply_shear(rotated)

    return skewed


def apply_shear(image: np.ndarray):
    """
    Applies shear transformation to the image.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Random shear factor
    factor = random.uniform(-0.5, 0.5)

    # Shear matrix :
    # [1, a, 0]
    # [0, 1, 0]
    # [0, 0, 1]
    # Define shear transformation matrix
    shear_matrix = np.float32([
        [1, factor, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Apply shear transformation
    sheared = cv2.warpPerspective(image, shear_matrix, (width, height))
    return sheared


def apply_crop(image: np.ndarray, scaling_ratio: float = 0.8):
    """
    Applies random scaling to the image.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    crop_width = int(width * scaling_ratio)
    crop_height = int(height * scaling_ratio)

    # Middle of the image
    x_offset = width // 2 - crop_width // 2
    y_offset = height // 2 - crop_height // 2

    # Apply scaling
    scaled = image[y_offset:y_offset + crop_height,
                   x_offset:x_offset + crop_width]

    # Resize back to original size
    scaled = cv2.resize(scaled, (width, height))

    return scaled


def apply_distortion(image: np.ndarray):
    """
    Applies distortion effects to the image.
    """
    # Convert to PIL Image for easier manipulation
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply random brightness and contrast adjustments
    brightness_factor = random.uniform(0.7, 1.3)
    contrast_factor = random.uniform(0.7, 1.3)

    # Random brightness effect
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)

    # Random contrast effect
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)

    # Apply blur effect
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Convert back to OpenCV format
    distorted = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return distorted


def display_augmentation(imgs: list[tuple[str, np.ndarray]]):
    """
    Displays the augmentation process in the console.
    """
    print("Applying data augmentation transformations...")
    _, axes = plt.subplots(2, 4, figsize=(15, 5 * 2))

    for i, (tr_name, img) in enumerate(imgs):
        row = i // 4
        col = i % 4
        # axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # axes[row, col].axis('off')
        # axes[row, col].set_title(tr_name, fontsize=10, fontweight='bold')
        axes[row, col].imshow(img, cmap='viridis', interpolation='nearest')
        axes[row, col].axis('off')
        axes[row, col].set_title(tr_name, fontsize=10, fontweight='bold')

    # Full screen and show
    plt.get_current_fig_manager().full_screen_toggle()
    plt.tight_layout()
    plt.show()
    plt.savefig("outputs/augmentation.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_augmented_image(image: np.ndarray, output_path: str):
    """
    Saves the augmented image to the specified path.
    """
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


def augment_image(image_path: str, display: bool = True):
    """
    Applies all 6 types of data augmentation to the given image.
    """
    # Load the original image
    original_image = load_image(image_path)

    if original_image is None:
        return

    # Get the directory and filename
    image_dir = os.path.dirname(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(os.path.basename(image_path))[1]

    transformations = {
        "Flip": apply_flip,
        "Rotate": apply_rotate,
        "Skew": apply_skew,
        "Shear": apply_shear,
        "Crop": apply_crop,
        "Distortion": apply_distortion
    }

    imgs = [(filename, original_image)]

    for tr_name, tr_function in transformations.items():
        transformed = tr_function(original_image)
        new_path = os.path.join(image_dir, f"{filename}_{tr_name}{ext}")
        save_augmented_image(transformed, new_path)
        imgs.append((f"{filename}_{tr_name}", transformed))

    if display:
        display_augmentation(imgs)


def main():
    """
    Main function to handle command line arguments and execute augmentation.
    """
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <path_to_image>")
        print("Ex: python Augmentation.py ./images/Apple_rust/image1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    # Check if it's a valid image file
    valid_extensions = ('.jpg', '.jpeg')
    if not image_path.lower().endswith(valid_extensions):
        print("Error: Provide a valid image file (.jpg, .jpeg).")
        sys.exit(1)

    # Remove background and save
    original = load_image(image_path)
    original = remove_background(original)
    save_augmented_image(original, image_path)

    # Apply data augmentation
    augment_image(image_path)

    print("All augmented images have been saved in the original directory.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
