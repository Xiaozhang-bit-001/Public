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
