#!/usr/bin/env python3
"""
Program for analyzing image distribution in a directory.
Generates pie charts and bar charts for each plant type.
"""

import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def compute_distribution(directory_path: str):
    """
    Analyzes the distribution of images in a directory and its subdirectories.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return None

    distribution = {}
    base_dir = Path(directory_path)

    # Browse subdirectories (maxdepth 2)
    for subdir in base_dir.iterdir():
        if subdir.is_dir() is False:
            continue

        category_name = subdir.name
        image_count = 0

        # Count .JPG images in the subdirectory
        for file_path in subdir.rglob('*.JPG'):
            if file_path.is_file():
                image_count += 1

        distribution[category_name] = image_count

    return distribution


def plot_distribution(distribution: dict, plant: str):
    """
    Creates pie charts and bar charts for image distribution.
    """
    if not distribution:
        print("No data to visualize.")
        return

    # Chart configuration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Image Distribution - {plant}',
                 fontsize=16, fontweight='bold')

    # Data for charts
    categories = list(distribution.keys())
    counts = list(distribution.values())

    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    ax1.pie(counts, labels=categories, autopct='%1.1f%%',
            colors=colors, startangle=90)

    # Bar chart
    bars = ax2.bar(categories, counts, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Categories', fontsize=12)
    ax2.set_ylabel('Number of images', fontsize=12)

    # Rotate labels for better readability
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    # Add values on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    # Tight layout and show
    plt.tight_layout()
    plt.show()
    plt.savefig("outputs/distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_summary(distribution: dict, plant: str):
    """
    Displays a summary of image distribution.
    """
    if not distribution:
        return

    total_images = sum(distribution.values())
    print(f"Total number of images: {total_images}")
    print(f"Number of categories: {len(distribution)}")
    print("\nDistribution by category:")
    print("-" * 50)

    # Sort by number of images descending
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1],
                         reverse=True)

    for category, count in sorted_dist:
        percentage = (count / total_images) * 100
        print(f"{category:<25} | {count:>6} images | {percentage:>5.1f}%")

    print("-" * 50)
    print(f"{'Total':<25} | {total_images:>6} images | 100.0%")


def main():
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <path_to_directory>")
        print("Example: python Distribution.py ./images")
        sys.exit(1)

    directory_path = sys.argv[1]

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        sys.exit(1)

    # Get plant type name (main directory name)
    plant = os.path.basename(os.path.abspath(directory_path))
    print(f"Directory analyzed: {os.path.abspath(directory_path)}")

    # Analyze distribution
    distribution = compute_distribution(directory_path)

    if distribution:
        # Display summary
        print_summary(distribution, plant)
        # Create visualizations
        plot_distribution(distribution, plant)
    else:
        print("No images found in the specified directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
