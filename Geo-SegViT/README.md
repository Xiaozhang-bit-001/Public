# Geo-SegViT: Remote Sensing Image Semantic Segmentation with Enhanced Cross-Layer Feature Fusion

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **Geo-SegViT**, a state-of-the-art semantic segmentation method for remote sensing imagery. Building upon the SegViT CNN-Transformer hybrid architecture, Geo-SegViT introduces learnable geographic spatial positional encoding, cross-layer channel-spatial dual attention fusion, and Swin-style pixel rearrangement upsampling to significantly improve segmentation accuracy and detail reconstruction.

> 📄 **Paper**: *A Geo-SegViT Remote Sensing Image Semantic Segmentation Method with Enhanced Cross-Layer Feature Fusion* (under review)  
> 👨‍💻 **Authors**: Junming Zhang, Fu Lv, Yongan Feng  
> 🏫 **Affiliation**: School of Software, Liaoning Technical University, Huludao, China

---

## 📊 Performance Highlights

| Dataset   | mIoU (%) | mPrec (%) | mRec (%) | mF1 (%) | OA (%) |
|-----------|----------|-----------|----------|---------|--------|
| Potsdam   | **88.24**| 93.81     | 93.58    | 93.69   | 94.79  |
| Vaihingen | **87.75**| 93.04     | 93.81    | 93.41   | 94.59  |

Geo-SegViT achieves state-of-the-art performance on the ISPRS Potsdam and Vaihingen benchmarks, consistently outperforming UNet, DeepLabV3+, UperNet, TransUNet, SwinUNet, HRViT-RS, DANet, and the baseline SegViT.

---

## 🧠 Key Innovations

1. **Learnable Geographic Spatial Position Encoding (GeoSpatialPosEncoding)**  
   Replaces fixed sinusoidal encoding with learnable x/y coordinate embeddings to better capture spatial-geographic characteristics of remote sensing images.

2. **Cross-Layer Channel-Spatial Dual Attention Fusion (CrossLayerFeatureFusion)**  
   Refines hierarchical features by applying cascaded channel and spatial attention after each encoder stage, suppressing noise and mitigating information attenuation.

3. **Land Prior Adaptive Fusion (LandPriorFusion)**  
   Enhances feature responses for small and weakly-featured targets (e.g., vehicles, bare soil) through learnable class-specific weight parameters.

4. **Swin-Style Inverse Window Pixel Rearrangement Upsampling (SwinUpsample)**  
   Eliminates checkerboard artifacts caused by transposed convolutions, achieving high-quality, artifact-free feature map restoration.

5. **Pure Transformer Decoder Block (DecoderTransformerBlock)**  
   Maintains architectural consistency with the encoder for improved feature fusion compatibility.

6. **Enhanced Depthwise Separable Segmentation Head (EnhancedSegHead)**  
   Improves edge detail refinement while significantly reducing parameter count compared to standard convolutions.

---

## 📁 Project Structure

```
Geo-SegViT/
│
├── models/                             # All model architectures
│   ├── Geo_SegViT.py                   # ★ Proposed Geo-SegViT model
│   ├── SegViT_RS.py                    # Baseline SegViT
│   ├── DANet.py                        # Dual Attention Network
│   ├── DeepLabVp.py                    # DeepLabV3+
│   ├── FCN.py                          # Fully Convolutional Network
│   ├── HRViT_RS.py                     # HRViT for remote sensing
│   ├── SwinUnet.py                     # Swin-Unet
│   ├── TransUnet.py                    # TransUNet
│   ├── UperNet.py                      # UPerNet
│   ├── modeling2.py                    # Shared modeling utilities
│   └── utils2.py                       # Loss functions (DiceLoss, etc.)
│
├── test/                               # Testing & evaluation scripts
│   ├── testPGeo_SegViT.py              # ★ Potsdam evaluation for Geo-SegViT
│   ├── testVGeo_SegViT.py              # ★ Vaihingen evaluation for Geo-SegViT
│   ├── testPDANet.py / testDANet.py    # DANet evaluation
│   ├── testPDeep.py / testVDeep.py     # DeepLabV3+ evaluation
│   ├── testPFCN.py / testVFCN.py       # FCN evaluation
│   ├── testPSwinUNet.py                # SwinUNet evaluation (Potsdam)
│   ├── testPTransUNet.py               # TransUNet evaluation (Potsdam)
│   ├── testPUPer.py                    # UPerNet evaluation (Potsdam)
│   ├── testTransUNet.py                # TransUNet evaluation
│   ├── testCC.py / testNew.py          # Utility test scripts
│   └── testPDANet.py                   # Additional DANet test
│
├── weights/                            # Pre-trained model weights
│   ├── Geo_SegViT/                     # ★ Geo-SegViT checkpoints
│   │   ├── bestP.pth                   #   Best on Potsdam (mIoU 88.24%)
│   │   └── bestV.pth                   #   Best on Vaihingen (mIoU 87.75%)
│   ├── DANet_P/ / DANet_V/            # DANet weights
│   ├── DeepLab_P/ / DeepLab_V/        # DeepLabV3+ weights
│   ├── DeeplabV3Plus_P/ / DeeplabV3Plus_V/
│   ├── FCN_P/ / FCN_V/                # FCN weights
│   ├── HRViTRS_Pots_256_256/          # HRViT-RS weights (Potsdam)
│   ├── HRViTRS_Vai_256_256/           # HRViT-RS weights (Vaihingen)
│   ├── SegViTRS_Pots_256_256/         # SegViT weights (Potsdam)
│   ├── SegViTRS_Vai_256_256/          # SegViT weights (Vaihingen)
│   ├── SwinUNet_P/ / SwinUNet_V/      # SwinUNet weights
│   ├── TransUNet_P/ / TransUNet_V/    # TransUNet weights
│   ├── UNet_P/ / UNet_V/              # UNet weights
│   └── UperNet_P/ / UperNet_V/        # UPerNet weights
│
├── Datasets/                           # Dataset configuration files
│   ├── Potsdam/
│   │   └── Geo_SegViT_Pots_config.json # Normalization params for Potsdam
│   └── Vaihingen/
│       └── Geo_SegViT_Vai_config.json  # Normalization params for Vaihingen
│
├── Test_Results/                       # Evaluation outputs
│   ├── Potsdam/
│   │   ├── Data/
│   │   ├── pred_maps/                  # Predicted segmentation maps
│   │   ├── visualization/              # Visual comparison images
│   │   │   ├── segvits_test_metrics.json
│   │   │   └── segvits_test_report.md
│   │   └── ...
│   └── Vaihingen/
│       ├── Data/
│       ├── pred_maps/
│       └── visualization/
│           ├── segvits_test_metrics.json
│           └── segvits_test_report.md
│
├── trainGeo_SegViT.py                  # ★ Main training script
├── create_npz3_RGB.py                  # Data preprocessing (raw → .npz)
├── dataset_synase2.py                  # Dataset class & augmentations
├── requirements.txt                    # Conda environment specification
├── .gitattributes                      # Git LFS configuration
└── README.md                           # This file
```

---

## ⚙️ Installation

### Requirements

- Python 3.9
- PyTorch 2.5.1
- CUDA 12.4 (recommended for GPU acceleration)
- Key packages: `einops`, `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `tqdm`, `tensorboardX`

### Setup

```bash
# Clone the repository
git clone https://github.com/Xiaozhang-bit-001/Public.git
cd Public/Geo-SegViT

# Create and activate conda environment (Windows)
conda create --name geo_segvit --file requirements.txt
conda activate geo_segvit

# For Linux/macOS, manually install core dependencies:
# pip install torch torchvision einops numpy opencv-python matplotlib scikit-learn tqdm tensorboardX
```

> **Note**: The provided `requirements.txt` is a platform-specific Conda explicit specification (win-64). Users on other platforms should install packages manually.

---

## 🚀 Usage

### Data Preparation

1. Download the ISPRS Potsdam and Vaihingen datasets from the [official ISPRS website](https://www.isprs.org/education/benchmarks.aspx).  
   **Only RGB orthophotos are used** in this work (near-infrared and DSM data are excluded).

2. Preprocess raw images into `.npz` format using the provided script:

```bash
python create_npz3_RGB.py --dataset potsdam --data_root /path/to/potsdam
python create_npz3_RGB.py --dataset vaihingen --data_root /path/to/vaihingen
```

Each `.npz` file will contain two arrays: `image` (3×256×256) and `label` (256×256).

### Training

Train Geo-SegViT using the main training script `trainGeo_SegViT.py`:

```bash
python trainGeo_SegViT.py \
    --root_path /path/to/npz_data \
    --list_dir /path/to/data_lists \
    --dataset potsdam \
    --max_epochs 150 \
    --base_lr 0.01 \
    --img_size 256
```

**Training Configuration:**
| Item | Setting |
|------|---------|
| Loss function | 0.5 × CrossEntropy + 0.5 × DiceLoss |
| Optimizer | SGD (momentum=0.9, weight_decay=0.0001) |
| Learning rate schedule | Polynomial decay (power=0.9) |
| Initial learning rate | 0.01 |
| Batch size | 8 |
| Max epochs | 150 |
| Random seed | 1234 |

The training script automatically:
- Logs metrics per epoch to both console and `log.txt`
- Saves epoch-wise detailed results to `train_results.csv` (loss, mIoU, mPrecision, mRecall, and per-class IoU/Precision/Recall)
- Saves **only the best model** (highest validation mIoU) as `best_model.pth`

### Evaluation

Before running, update the hardcoded paths in each test script (`ROOT_PATH`, `MODEL_WEIGHT_PATH`, `CONFIG_PATH`).

**Evaluate Geo-SegViT on Potsdam:**
```bash
python test/testPGeo_SegViT.py
```

**Evaluate Geo-SegViT on Vaihingen:**
```bash
python test/testVGeo_SegViT.py
```

The test scripts output:
- **Overall metrics**: OA, mIoU, mPrecision, mRecall, mF1
- **Per-class metrics**: IoU, Precision, Recall, F1 for all 6 classes
- **Confusion matrix** for detailed error analysis
- **Visualizations**: side-by-side comparison of input image, ground truth, and prediction
- **Markdown report** summarizing all results

---

## 📈 Pre-trained Weights

Pre-trained model weights corresponding to the paper's reported results:

| Model | Dataset | mIoU (%) | OA (%) | Weight File |
|-------|---------|----------|--------|-------------|
| **Geo-SegViT** | Potsdam | **88.24** | 94.79 | [`weights/Geo_SegViT/bestP.pth`](weights/Geo_SegViT/bestP.pth) |
| **Geo-SegViT** | Vaihingen | **87.75** | 94.59 | [`weights/Geo_SegViT/bestV.pth`](weights/Geo_SegViT/bestV.pth) |

Weights for all comparison models (UNet, DeepLabV3+, UperNet, TransUNet, SwinUNet, HRViT-RS, DANet, FCN, SegViT) are also available under the `weights/` directory.

---

## 📄 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhang2026geo,
  title={A Geo-SegViT Remote Sensing Image Semantic Segmentation Method with Enhanced Cross-Layer Feature Fusion},
  author={Zhang, Junming and Lv, Fu and Feng, Yongan},
  journal={Under review},
  year={2026}
}
```

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant Nos. 51874166, 52274206, 51904144). We sincerely thank the International Society for Photogrammetry and Remote Sensing (ISPRS) for providing the Potsdam and Vaihingen benchmark datasets.

---

## 🔗 Links

- **Repository**: [https://github.com/Xiaozhang-bit-001/Public](https://github.com/Xiaozhang-bit-001/Public)
- **ISPRS Benchmarks**: [https://www.isprs.org/education/benchmarks.aspx](https://www.isprs.org/education/benchmarks.aspx)
- **Original SegViT**: *SegViT: Semantic Segmentation with Plain Vision Transformers* (Zhang et al., NeurIPS 2022)

---

*For questions, collaborations, or issues, please open an issue on GitHub or contact the corresponding author.*
