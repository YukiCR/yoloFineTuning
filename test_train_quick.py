#!/usr/bin/env python3
"""
Quick test training script - just 3 epochs to verify setup
"""
import torch
from ultralytics import YOLO
from pathlib import Path

def quick_test_train():
    """Quick training test with minimal epochs"""
    print("🚀 Quick Training Test (3 epochs)")
    print("=" * 40)

    # Check prerequisites
    model_path = Path('models/yolov8s_pretrained.pt')
    data_path = Path('dataset/data.yaml')

    if not model_path.exists():
        print("❌ Pretrained model not found! Run setup_pretrained_model.py first.")
        return False

    if not data_path.exists():
        print("❌ Dataset not found! Run create_curated_dataset.py first.")
        return False

    # Load model
    print("Loading pretrained model...")
    model = YOLO(str(model_path))

    # Quick training config
    print("Starting quick test training (3 epochs)...")

    try:
        results = model.train(
            data=str(data_path),
            epochs=3,                    # Just 3 epochs for testing
            batch=4,                     # Smaller batch for quick test
            imgsz=640,
            optimizer='AdamW',
            lr0=0.001,
            device='cpu',                # CPU only
            workers=2,                   # Fewer workers
            verbose=True,
            save_period=1,               # Save every epoch
            project='results',
            name='quick_test',           # Separate from main training
            exist_ok=True,
            pretrained=True,

            # Minimal augmentation for quick test
            hsv_h=0.01,
            hsv_s=0.1,
            hsv_v=0.1,
            degrees=5.0,
            translate=0.05,
            scale=0.1,
            mosaic=0.0,                  # Disable mosaic for quick test
            mixup=0.0,                   # Disable mixup for quick test
            copy_paste=0.0,              # Disable copy-paste for quick test
        )

        print("\n✅ Quick test training completed successfully!")

        # Check results
        results_dir = Path('results/quick_test')
        if results_dir.exists():
            print(f"Results saved to: {results_dir}")

            # Check for final weights
            final_weights = results_dir / 'weights' / 'last.pt'
            if final_weights.exists():
                print(f"✅ Final weights saved: {final_weights}")

                # Test the trained model quickly
                print("\n🧪 Testing trained model...")
                test_image = list(Path('dataset/images/val').glob('*.jpg'))[0]
                test_results = model.predict(str(test_image), conf=0.25)
                print(f"Test predictions: {len(test_results[0].boxes)} detections")

                return True
            else:
                print("⚠️  Final weights not found")
                return False
        else:
            print("❌ Results directory not created")
            return False

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    success = quick_test_train()

    if success:
        print("\n🎉 Quick test passed! Ready for full training.")
        print("Run: python train_model.py for full 150-epoch training")
    else:
        print("\n❌ Quick test failed. Fix issues before full training.")
        exit(1)