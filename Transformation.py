#!/usr/bin/env python3
"""
Image transformation program for leaf analysis.
Implements 6 different image transformations for plant image processing.
"""

import os
import plantcv as pcv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, List

RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def error(message: str, parser: argparse.ArgumentParser):
    print(f"{RED}{BOLD}Error: {message}{RESET}")
    parser.print_help()
    return


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given path.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return None

    # Load image using plantcv
    image = pcv.readimage(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'.")
        return None

    return image


def mask_image(img):
    """
    Gives the black and white mask used in some transformations
    """
    gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    threshold = pcv.threshold.binary(gray_img=gray_img,
                                     threshold=60,
                                     max_value=255,
                                     object_type='dark')
    mask = pcv.invert(threshold)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    return mask


def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Applies Gaussian blur to the image.
    """
    return pcv.gaussian_blur(image, (15, 15), 0)


def process_directory(source: str, dest: str, transfos: list[str]):
    """
    Processes all images in a directory and saves transformations.
    """
    pass


def process_image(image: str, transfos: list[str]):
    """
    Processes a single image and saves transformations.
    """
    pass


def init_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python3 Transformation.py ./images/image1.jpg

  # Process directory
  python3 Transformation.py -s ./images -d ./output

  # Process directory with only some specified transformations
  python3 Transformation.py -s ./images -d ./output -t glossy,mask
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
        default='glossy,mask,roi,analysis,pseudolandmarks,colors',
        help='Transformations to apply'
    )
    return parser


def main():
    """
    Main function to handle command line arguments.
    """
    parser = init_parser()
    args = parser.parse_args()
    types = ['glossy', 'mask', 'roi', 'analysis', 'pseudolandmarks', 'colors']

    # if -src is provided, and -dst is not
    if args.source and not args.dest or not args.source and args.dest:
        error("Both dir must be provided", parser)
        return

    # if both -img and (-src and -dst) are provided
    if args.image and (args.source or args.dest):
        error("Either image or (-src and -dst) must be provided", parser)
        return

    # Check if transformations are valid
    args.transformations = args.transformations.split(',')
    for transformation in args.transformations:
        if transformation not in types:
            error(f"'{transformation}' is not valid\nUse: {types}", parser)
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
