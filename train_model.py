#!/usr/bin/env python3
"""
Fine-tune YOLOv8s on cat breed detection dataset
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import time
import matplotlib.pyplot as plt
import pandas as pd

def create_training_config():
    """Create training configuration for fine-tuning"""
    config = {
        # Model settings
        'model': 'models/yolov8s_pretrained.pt',  # Start from pretrained
        'data': 'dataset/data.yaml',              # Our dataset config
        'epochs': 150,                            # Sufficient for fine-tuning
        'batch': 32,                               # Conservative for CPU/memory
        'imgsz': 640,                             # Standard YOLO input size

        # Optimizer settings
        'optimizer': 'AdamW',                     # Good for fine-tuning
        'lr0': 0.001,                             # Low learning rate for fine-tuning
        'lrf': 0.01,                              # Final learning rate factor
        'weight_decay': 0.0005,                   # Regularization

        # Data augmentation (heavy for few-shot learning)
        'hsv_h': 0.015,      # Hue augmentation
        'hsv_s': 0.7,        # Saturation augmentation
        'hsv_v': 0.4,        # Value augmentation
        'degrees': 15.0,     # Rotation augmentation
        'translate': 0.1,    # Translation augmentation
        'scale': 0.5,        # Scaling augmentation
        'shear': 2.0,        # Shear augmentation
        'perspective': 0.0,  # Perspective augmentation
        'flipud': 0.0,       # Vertical flip
        'fliplr': 0.5,       # Horizontal flip
        'mosaic': 1.0,       # Mosaic augmentation (important for small datasets)
        'mixup': 0.2,        # Mixup augmentation
        'copy_paste': 0.3,   # Copy-paste augmentation

        # Training settings
        'device': 0,                              # Use GPU 0 (require GPU)
        'workers': 4,                             # Data loading workers
        'patience': 20,                           # Early stopping patience
        'save_period': 10,                        # Save checkpoint every N epochs
        'cache': False,                           # Don't cache images (memory)

        # Validation settings
        'val': True,                              # Validate during training
        'save_json': True,                        # Save results in JSON format
        'save_hybrid': False,                     # Don't save hybrid labels
        'conf': 0.25,                             # Confidence threshold
        'iou': 0.45,                              # IoU threshold for NMS

        # Output settings
        'project': 'results',                     # Output directory
        'name': 'cat_breed_detection',            # Experiment name
        'exist_ok': False,                        # Don't overwrite existing
        'pretrained': True,                       # Use pretrained weights
        'optimizer': 'AdamW',                     # Override default optimizer
        'verbose': True,                          # Verbose output
        'seed': 42,                               # Reproducibility
    }

    return config

def train_model():
    """Fine-tune YOLOv8s model on our dataset"""
    print("🚀 Starting YOLOv8s Fine-tuning")
    print("=" * 40)

    # Check if model exists - only use local pretrained model
    model_path = Path('models/yolov8s_pretrained.pt')
    if not model_path.exists():
        print("❌ Pretrained model not found at 'models/yolov8s_pretrained.pt'")
        print("❌ Please ensure the pretrained model file exists locally.")
        print("❌ This script does not download models from the internet.")
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")

    # Check if dataset exists
    data_path = Path('dataset/data.yaml')
    if not data_path.exists():
        print("❌ Dataset not found! Run create_curated_dataset.py first.")
        return False

    # Check GPU availability - REQUIRE GPU for intensive training
    if not torch.cuda.is_available():
        print("❌ GPU not available! This training requires GPU.")
        print("   Current setup: CPU only")
        print("   Please check CUDA installation and GPU drivers.")
        raise RuntimeError("GPU required for intensive training but not available")

    # Check GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {gpu_memory:.1f} GB")

    if gpu_memory < 6.0:  # Conservative requirement
        print(f"⚠️  Warning: GPU memory ({gpu_memory:.1f} GB) may be insufficient")
        print(f"   Recommended: ≥ 6 GB for stable training")

    # Load pretrained model - use local file only
    print(f"Loading pretrained model from: {model_path}")
    print("⚠️  Using local pretrained model only - no internet download")
    model = YOLO(str(model_path), task='detect')

    # Create training configuration
    config = create_training_config()

    print(f"Training configuration:")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Batch size: {config['batch']}")
    print(f"  - Image size: {config['imgsz']}")
    print(f"  - Learning rate: {config['lr0']}")
    print(f"  - Device: {config['device']}")

    # Start training
    print(f"\n⏱️  Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    try:
        results = model.train(**config)

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")

        # Save training results
        analyze_results(results)

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def analyze_results(results):
    """Analyze and visualize training results"""
    print("\n📊 Analyzing training results...")

    # Get results directory
    results_dir = Path('results/cat_breed_detection')
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return

    # Print final metrics
    print(f"Results saved to: {results_dir}")

    # List key files
    print("\nKey result files:")
    for file in results_dir.glob('*'):
        if file.is_file():
            print(f"  - {file.name}")

    # Check for best model
    best_model = results_dir / 'weights' / 'best.pt'
    if best_model.exists():
        print(f"\n🏆 Best model saved to: {best_model}")
    else:
        print("\n⚠️  Best model not found!")

    # Show training curves if available
    results_csv = results_dir / 'results.csv'
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            print(f"\nTraining data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Plot key metrics
            if 'mAP50' in df.columns:
                final_map50 = df['mAP50'].iloc[-1]
                print(f"Final mAP@50: {final_map50:.3f}")

        except Exception as e:
            print(f"Could not read results.csv: {e}")

if __name__ == "__main__":
    # Check system
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Start training
    success = train_model()

    if success:
        print("\n🎉 Fine-tuning completed successfully!")
        print("Next steps: Evaluate the model and create final report.")
    else:
        print("\n❌ Training failed. Check logs above.")
        exit(1)