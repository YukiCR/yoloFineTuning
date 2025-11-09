#!/usr/bin/env python3
"""
Create curated dataset for cat breed detection
Select 20 training + 5 validation images per breed
"""
import os
import shutil
import random
from pathlib import Path

# Define breeds and class mapping
BREEDS = ['persian', 'sphynx', 'ragdoll', 'scottish']
CLASS_MAPPING = {breed: idx for idx, breed in enumerate(BREEDS)}

# Dataset paths
SOURCE_DIR = Path('Cat-Breeds-Detection-Dataset')
TARGET_DIR = Path('dataset')

def create_curated_dataset():
    """Create curated dataset with 20 train + 5 val images per breed"""

    # Create target directories
    for split in ['train', 'val']:
        (TARGET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each breed
    for breed in BREEDS:
        print(f"Processing {breed}...")

        # Get all images for this breed
        train_images = list(SOURCE_DIR.glob(f'images/train/*{breed}*.jpg'))
        val_images = list(SOURCE_DIR.glob(f'images/val/*{breed}*.jpg'))

        print(f"  Found {len(train_images)} train, {len(val_images)} val images")

        # Randomly select 20 train + 5 val images
        selected_train = random.sample(train_images, min(20, len(train_images)))
        selected_val = random.sample(val_images, min(5, len(val_images)))

        print(f"  Selected {len(selected_train)} train, {len(selected_val)} val images")

        # Copy selected images and labels
        for split, selected_imgs in [('train', selected_train), ('val', selected_val)]:
            for img_path in selected_imgs:
                # Copy image
                target_img = TARGET_DIR / 'images' / split / img_path.name
                shutil.copy2(img_path, target_img)

                # Copy corresponding label
                label_name = img_path.stem + '.txt'
                source_label = SOURCE_DIR / 'labels' / split / label_name
                target_label = TARGET_DIR / 'labels' / split / label_name

                if source_label.exists():
                    # Update class ID in label file
                    with open(source_label, 'r') as f:
                        lines = f.readlines()

                    # Update class ID to match our mapping
                    updated_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Keep original class ID (assuming it's already 0-based per breed)
                            # But we'll remap to our class system
                            parts[0] = str(CLASS_MAPPING[breed])
                            updated_lines.append(' '.join(parts) + '\n')

                    with open(target_label, 'w') as f:
                        f.writelines(updated_lines)
                else:
                    print(f"  Warning: Label not found for {img_path.name}")

    # Create data.yaml file
    data_yaml = f"""
# Cat Breed Detection Dataset
path: {TARGET_DIR.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(BREEDS)}
names: {list(BREEDS)}
"""

    with open(TARGET_DIR / 'data.yaml', 'w') as f:
        f.write(data_yaml.strip())

    print(f"\nDataset created at: {TARGET_DIR}")
    print(f"Classes: {BREEDS}")
    print(f"Total images: {sum(1 for _ in TARGET_DIR.glob('images/*/*.jpg'))}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    create_curated_dataset()