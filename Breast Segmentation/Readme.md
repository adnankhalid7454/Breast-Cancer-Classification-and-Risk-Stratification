# BreastDivider - Breast MRI Segmentation

Automatic left-right breast segmentation from 3D MRI volumes using BreastDivider.

## 📦 Installation

```bash
pip install breastdivider
```

## 🚀 Quick Commands

### Single File Segmentation

```bash
breastdivider predict input.nii.gz output_seg.nii.gz --device cuda
```

### Using CPU

```bash
breastdivider predict input.nii.gz output_seg.nii.gz --device cpu
```

## 📊 Apply Segmentation Mask

```bash
# Create masked volume (removes background)
python3 << 'EOF'
import SimpleITK as sitk
import numpy as np

img = sitk.ReadImage("case_0000.nii.gz")
seg = sitk.ReadImage("case_0000_seg.nii.gz")

img_array = sitk.GetArrayFromImage(img)
seg_array = sitk.GetArrayFromImage(seg)

masked = img_array * seg_array

masked_img = sitk.GetImageFromArray(masked)
sitk.WriteImage(masked_img, "case_0000_masked.nii.gz")
print("Masked image saved!")
EOF
```

## 📂 Directory Structure

```
data/
├── breast_mri/              # Original images
│   ├── patient_001.nii.gz
│   ├── patient_002.nii.gz
│   └── ...
│
└── breast_seg/              # Segmentation masks
    ├── patient_001_seg.nii.gz
    ├── patient_002_seg.nii.gz
    └── ...
```

## 📋 Batch Processing Script

```bash
#!/bin/bash

INPUT_DIR="data/breast_mri"
OUTPUT_DIR="data/breast_seg"
DEVICE="cuda"

mkdir -p "$OUTPUT_DIR"

total=$(ls "$INPUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
count=0

for file in "$INPUT_DIR"/*.nii.gz; do
    count=$((count + 1))
    filename=$(basename "$file")
    outputname="${filename%.nii.gz}_seg.nii.gz"
    
    echo "[$count/$total] Processing $filename..."
    breastdivider predict "$file" "$OUTPUT_DIR/$outputname" --device "$DEVICE"
done

echo "✓ Segmentation complete!"
```

Save as `segment.sh` and run:

```bash
chmod +x segment.sh
./segment.sh
```

## 🎯 Common Commands

| Task | Command |
|------|---------|
| Install | `pip install breastdivider` |
| Segment (GPU) | `breastdivider predict input.nii.gz output.nii.gz --device cuda` |
| Segment (CPU) | `breastdivider predict input.nii.gz output.nii.gz --device cpu` |
| Download model | `breastdivider download` |
| Batch segment | `for f in *.nii.gz; do breastdivider predict "$f" "${f%.nii.gz}_seg.nii.gz" --device cuda; done` |

## 🔗 Resources

- **GitHub**: https://github.com/MIC-DKFZ/BreastDivider
- **Model**: https://huggingface.co/ykirchhoff/BreastDividerModel
- **Dataset**: https://huggingface.co/datasets/Bubenpo/BreastDividerDataset

## 📊 Model Info

- **Training Data**: 17,956 MRI scans
- **Accuracy**: 0.99 Dice score
- **Output**: Binary segmentation masks (breast tissue)

## ⚠️ Troubleshooting

```bash
# Model download issues
breastdivider download

# Check if GPU is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Use CPU if GPU memory limited
breastdivider predict input.nii.gz output.nii.gz --device cpu
```

## 📝 Citation

```bibtex
@article{rokuss2025breastdivider,
  title     = {Divide and Conquer: A Large-Scale Dataset and Model for Left–Right Breast MRI Segmentation},
  author    = {Rokuss, Maximilian and Hamm, Benjamin and Kirchhoff, Yannick and Maier-Hein, Klaus},
  journal   = {arXiv preprint arXiv:2507.13830},
  year      = {2025}
}
```


