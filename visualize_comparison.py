#!/usr/bin/env python3
"""
Visualize comparison between pretrained YOLOv8s and fine-tuned model
on randomly selected validation images from each class

how to run this script:
```bash
python visualize_comparison.py --pretrained models/yolov8s_pretrained.pt --finetuned results/cat_breed_detection/weights/best.pt --data dataset/data.yaml --val-dir dataset --output model_comparison.png --conf 0.60 --seed 55
```
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import argparse
import yaml

def load_class_names(data_yaml_path):
    """Load class names from data.yaml"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def find_validation_images(val_dir, class_names):
    """Find validation images for each class"""
    val_dir = Path(val_dir)
    images_dir = val_dir / 'images' / 'val'
    labels_dir = val_dir / 'labels' / 'val'

    class_images = {i: [] for i in range(len(class_names))}

    # Find all validation images
    for img_file in images_dir.glob('*.jpg'):
        # Look for corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id in class_images:
                        class_images[class_id].append(img_file)

    return class_images

def select_random_images(class_images, num_per_class=1):
    """Select random images for each class"""
    selected = {}
    for class_id, images in class_images.items():
        if images:
            selected[class_id] = random.sample(images, min(num_per_class, len(images)))
        else:
            selected[class_id] = []
    return selected

def run_inference(model, image_path, conf=0.25):
    """Run inference on a single image"""
    results = model(image_path, conf=conf)
    return results[0]  # Return first (and only) result

def create_comparison_plot(pretrained_results, finetuned_results, class_names, save_path=None):
    """Create comparison plot showing results from both models"""
    num_classes = len(class_names)
    fig, axes = plt.subplots(2, num_classes, figsize=(5*num_classes, 10))

    if num_classes == 1:
        axes = axes.reshape(2, 1)

    # Plot pretrained model results (top row)
    for i, (class_id, result) in enumerate(pretrained_results.items()):
        if result is not None:
            ax = axes[0, i]
            plot_result(ax, result, f"Pretrained - {class_names[class_id]}")
        else:
            ax = axes[0, i]
            ax.text(0.5, 0.5, f"No detection\n{class_names[class_id]}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Pretrained - {class_names[class_id]}")

    # Plot fine-tuned model results (bottom row)
    for i, (class_id, result) in enumerate(finetuned_results.items()):
        if result is not None:
            ax = axes[1, i]
            plot_result(ax, result, f"Fine-tuned - {class_names[class_id]}")
        else:
            ax = axes[1, i]
            ax.text(0.5, 0.5, f"No detection\n{class_names[class_id]}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Fine-tuned - {class_names[class_id]}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    return fig

def plot_result(ax, result, title):
    """Plot detection result on axis"""
    # Get the plotted image with boxes
    plotted_img = result.plot()
    ax.imshow(plotted_img)
    ax.set_title(title)
    ax.axis('off')

    # Add detection count
    # if hasattr(result, 'boxes') and len(result.boxes) > 0:
    #     num_detections = len(result.boxes)
    #     ax.text(0.02, 0.98, f'Detections: {num_detections}',
    #            transform=ax.transAxes, ha='left', va='top',
    #            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    parser = argparse.ArgumentParser(description='Compare pretrained vs fine-tuned YOLO models')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained model (yolov8s.pt)')
    parser.add_argument('--finetuned', type=str, required=True,
                       help='Path to fine-tuned model')
    parser.add_argument('--data', type=str, default='dataset/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--val-dir', type=str, default='dataset/val',
                       help='Path to validation directory')
    parser.add_argument('--output', type=str, default='model_comparison.png',
                       help='Output path for comparison plot')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detection')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("🖼️  Starting Model Comparison Visualization")
    print("=" * 50)

    # Check if models exist
    pretrained_path = Path(args.pretrained)
    finetuned_path = Path(args.finetuned)

    if not pretrained_path.exists():
        print(f"❌ Pretrained model not found: {pretrained_path}")
        return

    if not finetuned_path.exists():
        print(f"❌ Fine-tuned model not found: {finetuned_path}")
        return

    # Load class names
    class_names = load_class_names(args.data)
    print(f"📋 Loaded {len(class_names)} classes: {class_names}")

    # Find validation images
    class_images = find_validation_images(args.val_dir, class_names)

    # Check if we have images for each class
    for class_id, images in class_images.items():
        if not images:
            print(f"⚠️  No validation images found for class: {class_names[class_id]}")

    # Select random images
    selected_images = select_random_images(class_images, num_per_class=1)

    # Load models
    print("⏳ Loading pretrained model...")
    pretrained_model = YOLO(str(pretrained_path))

    print("⏳ Loading fine-tuned model...")
    finetuned_model = YOLO(str(finetuned_path))

    # Run inference
    pretrained_results = {}
    finetuned_results = {}

    print("🔍 Running inference on selected images...")
    for class_id, image_paths in selected_images.items():
        if image_paths:
            image_path = image_paths[0]
            print(f"  Processing {class_names[class_id]}: {image_path.name}")

            # Run both models
            pretrained_result = run_inference(pretrained_model, image_path, args.conf)
            finetuned_result = run_inference(finetuned_model, image_path, args.conf)

            pretrained_results[class_id] = pretrained_result
            finetuned_results[class_id] = finetuned_result

    # Create comparison plot
    print("📊 Creating comparison plot...")
    fig = create_comparison_plot(pretrained_results, finetuned_results, class_names, args.output)

    # Display plot
    plt.show()

    print(f"✅ Comparison completed!")
    print(f"📈 Results saved to: {args.output}")

if __name__ == "__main__":
    main()