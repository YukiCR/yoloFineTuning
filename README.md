# Cat Breed Detection with YOLOv8

A object detection project implementing YOLOv8s for cat breed classification and localization with limited data and resources. See `main.pdf` for a Chinese homework report.

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU
- Conda (recommended)

## Environment Setup

### Step 1: Create Conda Environment

```bash
conda create -n cat_detection python=3.8
conda activate cat_detection
```

### Step 2: Install PyTorch with CUDA Support

```bash
# For CUDA 12.4, refer to https://pytorch.org/get-started/locally/ for your device
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Check PyTorch installation:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 3: Install Ultralytics and Dependencies

```bash
# Install ultralytics (better compatibility via conda-forge)
conda install -c conda-forge ultralytics

# Alternative: pip install ultralytics
```
also refer to `requirements_clean.txt` for version of packages we used

## Usage Guide

### Step 1: Download Dataset

```bash
git clone https://github.com/AWREDD/Cat-Breeds-Detection-Dataset.git
```

### Step 2: Create Curated Subset (Optional)

Create a smaller curated dataset with 20 train + 5 val images per breed (4 selected breeds):

```bash
python create_curated_dataset.py
```

This creates:
- `dataset/images/{train,val}/` - Selected images
- `dataset/labels/{train,val}/` - Corresponding YOLO format annotations
- `dataset/data.yaml` - Dataset configuration file

Visualize the curated dataset:
```bash
python visualize_dataset.py
```

### Step 3: Setup Pretrained Model

```bash
python setup_pretrained_model.py
```

Downloads YOLOv8s pretrained on COCO dataset and saves to `models/yolov8s_pretrained.pt`.

### Step 4: Test Setup and Train

**Option A: Quick Test (Recommended First)**
Run a quick 3-epoch test to verify setup:

```bash
python test_gpu_setup.py
python test_train_quick.py
```

**Option B: Full Training**
Train the model (150 epochs, ~45-60 minutes):

```bash
python train_model.py
```

Training configuration:
- Epochs: 150
- Batch size: 32
- Optimizer: AdamW
- Heavy augmentation (mosaic, copy-paste, rotation, color transforms)
- Results saved to `results/cat_breed_detection/`
- Best model weights: `results/cat_breed_detection/weights/best.pt`

### Step 5: Compare Models

Compare pretrained vs fine-tuned models on validation images:

```bash
python visualize_comparison.py \
    --pretrained models/yolov8s_pretrained.pt \
    --finetuned results/cat_breed_detection/weights/best.pt \
    --data dataset/data.yaml \
    --val-dir dataset \
    --output model_comparison.png \
    --conf 0.60 \
    --seed 55
```

This generates a side-by-side comparison plot showing detections from both models.

## Project Structure

```
cat-breed-detection/
├── Cat-Breeds-Detection-Dataset/    # Original dataset (git clone)
├── dataset/                         # Curated subset (generated)
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── data.yaml
├── models/                          # Pretrained weights (yolov8s_pretrained.pt)
├── results/                         # Training outputs
│   └── cat_breed_detection/
│       └── weights/{best.pt, last.pt}
├── MyZJUreportTemplate/            # LaTeX report project
├── *.py                            # Python scripts
├── requirements_clean.txt          # Python dependencies
├── README.md                       # This file

```

## Documentation

- **main.pdf**: Comprehensive Chinese project report documenting methodology, implementation, and results
- **plan.md**: Detailed technical decisions and implementation phases
- **CLAUDE.md**: Project context and constraints (for AI assistance)

## Key Files

- `create_curated_dataset.py` - Dataset curation script
- `visualize_dataset.py` - Dataset visualization
- `setup_pretrained_model.py` - Download and setup pretrained YOLOv8s
- `test_gpu_setup.py` - GPU/CUDA diagnostic tool
- `test_train_quick.py` - Quick 3-epoch training test
- `train_model.py` - Main training script (150 epochs)
- `visualize_comparison.py` - Model comparison visualization

## Troubleshooting

### GPU Issues
If `torch.cuda.is_available()` returns False:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA installation
3. Reinstall PyTorch with correct CUDA version

See `test_gpu_setup.py` for detailed diagnostics and suggested fixes.

### Out of Memory
If training runs out of GPU memory:
- Reduce batch size in `train_model.py`
- Close other GPU applications

### Model Not Found
Ensure pretrained model exists before training:
```bash
python setup_pretrained_model.py
```

## Results

Expected training results:
- **Best model**: `results/cat_breed_detection/weights/best.pt`

## Citation

+ Dataset: [Cat Breeds Detection Dataset (AWREDD)](https://github.com/AWREDD/Cat-Breeds-Detection-Dataset.git)
+ Model: YOLOv8 by Ultralytics

---

**Note**: 
+ This is a homework project focused on methodology and implementation under constraints. The limited dataset (20 samples/class) represents an ultra-low data regime where careful fine-tuning and augmentation strategies are essential.
+ Some parts of this project are finished with the help of AI
