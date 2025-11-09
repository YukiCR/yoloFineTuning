#!/usr/bin/env python3
"""
Test GPU setup and CUDA availability for training
"""
import torch
import subprocess
import sys
from pathlib import Path

def test_gpu_setup():
    """Comprehensive GPU setup test"""
    print("🔍 Testing GPU Setup")
    print("=" * 40)

    # Basic PyTorch CUDA test
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA not available through PyTorch")
        print("\nDiagnostic information:")

        # Check if nvidia-smi works
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ nvidia-smi works (NVIDIA driver installed)")
                print("   Output:", result.stdout.split('\n')[0])
            else:
                print("❌ nvidia-smi failed")
        except FileNotFoundError:
            print("❌ nvidia-smi not found (NVIDIA driver not installed)")

        # Check CUDA environment variables
        import os
        cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
        for var in cuda_vars:
            value = os.environ.get(var, 'Not set')
            print(f"   {var}: {value}")

        return False

    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")

        # Memory check
        if memory_gb < 6.0:
            print(f"  ⚠️  Warning: Low memory for intensive training")
        else:
            print(f"  ✅ Sufficient memory for training")

    # Test GPU memory allocation
    print("\n🧪 Testing GPU memory allocation...")
    try:
        # Allocate small tensor
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"✅ GPU memory allocation successful")
        print(f"   Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        # Test computation
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ GPU computation successful")

        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ GPU memory/computation test failed: {e}")
        return False

    print("\n✅ All GPU tests passed!")
    return True

def suggest_fixes():
    """Suggest fixes for common GPU issues"""
    print("\n🔧 Suggested fixes for GPU issues:")
    print("1. Check NVIDIA driver installation:")
    print("   sudo ubuntu-drivers devices")
    print("   sudo ubuntu-drivers autoinstall")
    print()
    print("2. Install CUDA toolkit:")
    print("   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin")
    print("   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600")
    print("   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub")
    print("   sudo add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /\"")
    print("   sudo apt-get update")
    print("   sudo apt-get -y install cuda")
    print()
    print("3. Install PyTorch with CUDA support:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("4. Set CUDA environment variables:")
    print("   export CUDA_HOME=/usr/local/cuda")
    print("   export PATH=$CUDA_HOME/bin:$PATH")
    print("   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH")

if __name__ == "__main__":
    success = test_gpu_setup()

    if success:
        print("\n🎉 GPU setup is ready for intensive training!")
        print("You can now run: python train_model.py")
    else:
        print("\n❌ GPU setup has issues. Please fix before training.")
        suggest_fixes()
        sys.exit(1)