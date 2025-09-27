#!/usr/bin/env python3
"""
Image transformation program for leaf analysis.
Implements 6 different image transformations for plant image processing.
"""

import os
import fnmatch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.PlantCV import ImageProcessor
from utils.error import error


def process_directory(source: str, dest: str, transfos: list[str]):
    """
    Processes all images in a directory and saves transformations.
    """
    os.makedirs(dest, exist_ok=True)
    for img_path in fnmatch.filter(os.listdir(source), '*.JPG'):
        print(f"⏳ Processing image: {img_path}...", end='', flush=True)
        processor = ImageProcessor(os.path.join(source, img_path), transfos)
        for transformation in transfos:
            path = os.path.join(dest, img_path.replace(".JPG", "_") + transformation + ".JPG")
            cv2.imwrite(path, processor.get_transformation(transformation))
        print(f"\r✅ Processed image: {img_path}    ")


def set_img(axes, row, col, img, title):
    axes[row, col].imshow(img, cmap='viridis', interpolation='nearest')
    axes[row, col].axis('off')
    axes[row, col].set_title(title, fontsize=10, fontweight='bold')


def process_image(image_path: str, transfos: list[str]):
    """
    Processes a single image and displays transformations in a window.
    """
    processor = ImageProcessor(image_path, transfos)

    # Init the plot figure and axes
    cols = 3
    rows = 2
    _, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Display the original image
    set_img(axes, 0, 0, processor.original, 'Original')

    row = 0
    col = 1
    # Set all images transformations in the plot
    for i, transformation in enumerate(transfos):
        row = (i + 1) // cols
        col = (i + 1) % cols
        transformed_img = processor.get_transformation(transformation)
        title = processor.get_transformation_title(transformation)
        set_img(axes, row, col, transformed_img, title)

    # Full screen and show
    plt.get_current_fig_manager().full_screen_toggle()
    plt.tight_layout()
    plt.show()


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python3 Transformation.py ./images/image1.jpg

  # Process directory
  python3 Transformation.py -s ./images -d ./output

  # Process directory with only some specified transformations
  python3 Transformation.py -s ./images -d ./output -t gaussian,mask
""")

    parser.add_argument('image', nargs='?', help='Path to single image')
    parser.add_argument(
        '-s', '--source',
        help='Source directory containing images'
    )
    parser.add_argument(
        '-d', '--dest',
        help='Destination directory for saved transformations'
    )
    parser.add_argument(
        '-t', '--transformations',
        default='gaussian,mask,roi,analysis,pseudolandmarks',
        help='Transformations to apply'
    )
    return parser


def main():
    """
    Main function to handle command line arguments.
    """
    parser = create_parser()
    args = parser.parse_args()
    types = ['gaussian', 'mask', 'roi', 'analysis', 'pseudolandmarks']

    # if -src is provided, and -dst is not
    if args.source and not args.dest or not args.source and args.dest:
        error("Both dir must be provided")
        return

    # if both -img and (-src and -dst) are provided
    if args.image and (args.source or args.dest):
        error("Either image or (-src and -dst) must be provided")
        return

    # Check if transformations are valid
    args.transformations = args.transformations.split(',')
    for transformation in args.transformations:
        if transformation not in types:
            error(f"'{transformation}' is not valid\nUse: {types}")
            return

    print(f"Applying transformations: {args.transformations}")

    # Process single image
    if args.image:
        process_image(args.image, args.transformations)

    # Process directory
    elif args.source and args.dest:
        process_directory(args.source, args.dest, args.transformations)


if __name__ == "__main__":
    main()
