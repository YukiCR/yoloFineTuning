#!/usr/bin/env python3
"""
Download and test YOLOv8s pretrained model
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

def download_pretrained_model():
    """Download YOLOv8s pretrained model"""
    print("Downloading YOLOv8s pretrained model...")

    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    try:
        # Load pretrained model (this will download if not exists)
        model = YOLO('yolov8s.pt')

        # Save model to local directory
        model_path = models_dir / 'yolov8s_pretrained.pt'
        model.save(str(model_path))

        print(f"✅ Model downloaded and saved to: {model_path}")
        print(f"Model info:")
        print(f"  - Task: {model.task}")
        print(f"  - Classes: {len(model.names)}")
        print(f"  - Model type: {model.ckpt.get('model_type', 'Unknown')}")

        return model, model_path

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None, None

def test_pretrained_model(model):
    """Test pretrained model on a sample image"""
    print("\n🧪 Testing pretrained model...")

    # Get a random image from our dataset
    dataset_dir = Path('dataset')
    train_images = list(dataset_dir.glob('images/train/*.jpg'))

    if not train_images:
        print("❌ No test images found in dataset!")
        return False

    # Select a random image
    test_image = random.choice(train_images)
    print(f"Testing on: {test_image.name}")

    try:
        # Run inference
        results = model.predict(source=str(test_image), conf=0.25, iou=0.45)

        # Print results
        print(f"✅ Inference completed!")
        print(f"  - Detections: {len(results[0].boxes)}")

        if len(results[0].boxes) > 0:
            print("  - Detected objects:")
            for box in results[0].boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                print(f"    - {class_name}: {confidence:.3f}")

        # Visualize results
        visualize_prediction(test_image, results[0], model.names)

        return True

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def visualize_prediction(image_path, result, class_names):
    """Visualize prediction results"""
    # Load original image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Image with predictions
    annotated_image = result.plot()  # This returns annotated image
    ax2.imshow(annotated_image)
    ax2.set_title('YOLOv8s Predictions', fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()

    # Save visualization
    output_path = 'pretrained_test_result.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_path}")

    plt.show()

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        print(f"\n📊 GPU Memory Info:")
        print(f"  - Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  - Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"  - Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    else:
        print("\n⚠️  Running on CPU (no GPU detected)")

if __name__ == "__main__":
    import random

    print("🚀 YOLOv8s Pretrained Model Setup")
    print("=" * 40)

    # Check GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download model
    model, model_path = download_pretrained_model()

    if model:
        # Check GPU memory after loading model
        check_gpu_memory()

        # Test model
        success = test_pretrained_model(model)

        if success:
            print("\n✅ Model setup and testing completed successfully!")
            print("Ready for fine-tuning phase!")
        else:
            print("\n⚠️  Model downloaded but testing failed.")
    else:
        print("\n❌ Failed to download pretrained model.")
        exit(1)