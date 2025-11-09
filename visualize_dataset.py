#!/usr/bin/env python3
"""
Visualize dataset samples with bounding boxes
Randomly select 1 sample from each breed and create visualization
"""
import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

# Dataset configuration
DATASET_DIR = Path('dataset')
BREEDS = ['persian', 'sphynx', 'ragdoll', 'scottish']
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Different colors for each breed

def load_image_and_labels(image_path, label_path):
    """Load image and corresponding YOLO format labels"""
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Load labels
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, w, h = map(float, parts[:5])
                    # Convert from normalized YOLO format to pixel coordinates
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    boxes.append([x1, y1, x2, y2, int(class_id)])

    return image, boxes, width, height

def create_visualization():
    """Create visualization with one random sample from each breed"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, breed in enumerate(BREEDS):
        print(f"Processing {breed}...")

        # Get all training images for this breed
        train_images = list(DATASET_DIR.glob(f'images/train/*{breed}*.jpg'))

        if not train_images:
            print(f"No images found for {breed}")
            continue

        # Randomly select one image
        selected_image = random.choice(train_images)

        # Get corresponding label file
        label_path = DATASET_DIR / 'labels' / 'train' / (selected_image.stem + '.txt')

        # Load image and labels
        image, boxes, width, height = load_image_and_labels(selected_image, label_path)

        # Plot image
        axes[idx].imshow(image)
        axes[idx].set_title(f'{breed.title()} Cat', fontsize=14, fontweight='bold')

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=COLORS[idx],
                                   facecolor='none', linestyle='--')
            axes[idx].add_patch(rect)

            # Add confidence text (just show breed name)
            axes[idx].text(x1, y1-5, breed.title(), fontsize=10,
                          color=COLORS[idx], fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Remove axis ticks
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

        # Add grid for better visualization
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.suptitle('Cat Breed Detection Dataset - Sample Images', fontsize=16, fontweight='bold', y=0.98)

    # Save the figure
    output_path = 'dataset_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Visualization saved as: {output_path}")

    # Also display basic stats
    print("\nDataset Statistics:")
    for breed in BREEDS:
        train_count = len(list(DATASET_DIR.glob(f'images/train/*{breed}*.jpg')))
        val_count = len(list(DATASET_DIR.glob(f'images/val/*{breed}*.jpg')))
        print(f"  {breed}: {train_count} train, {val_count} val")

    plt.show()

    return output_path

if __name__ == "__main__":
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"Dataset directory {DATASET_DIR} not found!")
        exit(1)

    # Create visualization
    output_file = create_visualization()
    print(f"\nVisualization complete! Check {output_file}")