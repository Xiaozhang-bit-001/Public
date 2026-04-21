import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import json
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ======================== 1. 核心配置（与模型/训练脚本对齐） ========================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据集配置
DATASET_NAME = 'Pots_256'
ROOT_PATH = r'G:\erya308\zhangjunming\Geo-SegViT\datasets\Potsdam\npz_data_RGB_improved'
MODEL_WEIGHT_PATH = r"G:\erya308\zhangjunming\modeL\Geo-SegViT\ComparedModel2_Best\SegViTRS_Improved_Pot_networks\SegViTRS_Improved_Pots_256_256\iter30k_epo150_bs8_lr0.01_s1234\best_model.pth"
CONFIG_PATH = r"G:\erya308\zhangjunming\Geo-SegViT\Dataset\Potsdam\Geo_SegViT_Pots_config.json"

# 超参数（与模型/训练一致）
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_CLASSES = 6
IN_CHANNELS = 3
VIS_NUM = 3000
DEBUG = True

# 输出配置
OUTPUT_DIR = r'G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\TestResults\Geo-SegViT_P'

# 类别配置
CLASS_NAMES = [
    'Low shrub', 'Impervious surfaces', 'Building',
    'Background', 'Vegetation', 'Vehicle'
]
VALID_CLASSES = [0, 1, 2, 3, 4, 5]
CLASS_COLORS = [
    (0, 255, 255), (255, 255, 255), (0, 0, 255),
    (255, 0, 0), (0, 255, 0), (255, 255, 0)
]

# ======================== 2. 数据加载（与原测试完全一致） ========================
class SegViTRSTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, img_size=256, transform=None):
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform

        self.file_paths = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_paths.append(os.path.join(root_path, f))

        print(f"📁 加载{len(self.file_paths)}个测试样本")

        # 调试：统计全数据集标签范围
        if DEBUG:
            all_labels = []
            sample_num = min(500, len(self.file_paths))
            for i in range(sample_num):
                data = np.load(self.file_paths[i])
                all_labels.extend(np.unique(data['label']))
            all_labels = np.unique(all_labels)
            print(f"🔍 前500个样本的标签范围: {all_labels}")
            print(f"🔍 包含的类别: {[CLASS_NAMES[int(l)] for l in all_labels]}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        file_name = os.path.basename(npz_path).replace('.npz', '')

        # 加载数据
        data = np.load(npz_path)
        img = data['image']  # (3, 256, 256) → NumPy数组
        mask = data['label']  # (256, 256) → NumPy数组

        # 验证维度
        assert img.shape == (3, self.img_size, self.img_size), f"图像维度错误: {img.shape}"
        assert mask.shape == (self.img_size, self.img_size), f"标签维度错误: {mask.shape}"

        # 归一化
        if self.transform is not None:
            img_trans = img.transpose(1, 2, 0)  # (3,H,W) → (H,W,3)
            img_trans = self.transform(img_trans)
            img = img_trans.transpose(2, 0, 1)  # (H,W,3) → (3,H,W)

        # 调试：输出第一个样本信息
        if DEBUG and idx == 0:
            unique_labels = np.unique(mask)
            print(f"\n🔍 样本{file_name}标签分布: {unique_labels}")
            print(f"🔍 输入图像像素范围: [{img.min():.2f}, {img.max():.2f}]")
            print(f"🔍 标签值范围: [{mask.min()}, {mask.max()}]")

        # 转换为tensor
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name

# ======================== 3. 评价指标计算（完全修复版） ========================
class SegViTRSMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        for pred, target in zip(preds, targets):
            mask = (target >= 0) & (target < self.num_classes)
            pred_valid = pred[mask]
            target_valid = target[mask]

            if len(pred_valid) > 0 and len(target_valid) > 0:
                self.confusion_matrix += confusion_matrix(
                    target_valid, pred_valid,
                    labels=list(range(self.num_classes))
                )

    def compute(self):
        cm = self.confusion_matrix.copy()
        results = {}

        # 每类指标计算
        iou = []
        precision = []
        recall = []
        f1 = []
        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp

            iou_cls = tp / (tp + fp + fn + 1e-8)
            precision_cls = tp / (tp + fp + 1e-8)
            recall_cls = tp / (tp + fn + 1e-8)
            f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-8)

            iou.append(iou_cls)
            precision.append(precision_cls)
            recall.append(recall_cls)
            f1.append(f1_cls)

        # 平均指标
        valid_mask = np.array([cls in VALID_CLASSES for cls in range(self.num_classes)])
        results['mIoU'] = float(np.mean(np.array(iou)[valid_mask]))
        results['mPrecision'] = float(np.mean(np.array(precision)[valid_mask]))
        results['mRecall'] = float(np.mean(np.array(recall)[valid_mask]))
        results['mF1'] = float(np.mean(np.array(f1)[valid_mask]))

        # 总体精度OA
        total_tp = np.diag(cm).sum()
        total_samples = cm.sum()
        results['OA'] = float(total_tp / (total_samples + 1e-8))

        # ✅ 修复：全部加上 enumerate
        results['per_class_iou'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(iou)}
        results['per_class_precision'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(precision)}
        results['per_class_recall'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(recall)}
        results['per_class_f1'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(f1)}

        # 调试：输出混淆矩阵
        if DEBUG:
            print("\n🔍 混淆矩阵:")
            print(cm)

        return results

# ======================== 4. 可视化函数（完全不变） ========================
def vis_segvitrs_result(img, mask, pred, file_name, save_path, mean, std):
    # 反归一化
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean

    if mean.max() > 1:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # 标签转彩色图
    def mask2color(mask_data):
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
        h, w = mask_data.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in range(len(CLASS_COLORS)):
            color_img[mask_data == cls] = CLASS_COLORS[cls]
        return color_img

    mask_color = mask2color(mask)
    pred_color = mask2color(pred)

    # 绘图保存
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('真实标签', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('SegViTRS预测结果', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(file_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== 5. 主函数（完全一致） ========================
def main():
    # 添加项目根目录
    PROJECT_ROOT = r'G:\erya308\zhangjunming\Geo-SegViT'
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        print(f"✅ 添加项目根目录: {PROJECT_ROOT}")

    # 创建输出目录
    test_output_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    vis_dir = os.path.join(test_output_dir, 'visualization')
    pred_map_dir = os.path.join(test_output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)
    print(f"📌 测试结果保存路径: {test_output_dir}")

    # 加载训练配置和归一化参数
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    train_mean = np.array(train_config['dataset_stats']['mean'])
    train_std = np.array(train_config['dataset_stats']['std'])
    print(f"📊 归一化参数 - 均值: {train_mean.round(4)}, 标准差: {train_std.round(4)}")

    # 数据预处理
    def transform(img):
        img = img.astype(np.float32)
        if train_mean.max() > 1:
            img = (img - train_mean) / train_std
        else:
            img = img / 255.0
            img = (img - train_mean) / train_std
        return img

    # 加载数据集
    test_dataset = SegViTRSTestDataset(
        root_path=ROOT_PATH,
        img_size=IMG_SIZE,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    # ===================== 模型直接使用 =====================
    print("\n🔧 加载 SegViTRS_Improved 模型...")
    model = SegViTRS_Improved().to(DEVICE)

    # 加载权重
    try:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)

    # 处理多GPU权重
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 加载权重
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ 权重加载成功（strict=True）")
    except RuntimeError as e:
        print(f"⚠️ 权重严格加载失败: {e}")
        print("🔄 尝试非严格加载...")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # 验证模型输出
    print("\n🔍 验证模型权重有效性...")
    test_input = torch.randn(1, 3, 256, 256).to(DEVICE)
    with torch.no_grad():
        test_output = model(test_input)
    output_mean = test_output.mean().item()
    output_std = test_output.std().item()
    print(f"随机输入的模型输出均值: {output_mean:.4f}, 标准差: {output_std:.4f}")
    if output_std < 1e-3:
        print("❌ 模型权重异常！输出无方差")
    else:
        print("✅ 模型权重验证通过")

    # 初始化指标
    metrics = SegViTRSMetrics(NUM_CLASSES)
    metrics.reset()

    # 开始测试
    print("\n🚀 开始测试 SegViTRS_Improved 模型...")
    vis_count = 0
    with torch.no_grad():
        for batch_idx, (imgs, masks, file_names) in enumerate(tqdm(test_loader, desc='SegViTRS测试进度')):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 前向推理
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            # 调试信息
            if DEBUG and batch_idx == 0:
                unique_preds = np.unique(preds.cpu().numpy())
                print(f"🔍 第一个batch预测类别: {unique_preds}")
                output_probs = F.softmax(outputs[0], dim=0).cpu().numpy()
                print(f"🔍 第一个样本预测概率: {[round(output_probs[i].mean(), 4) for i in range(6)]}")

            # 更新指标
            metrics.update(preds, masks)

            # 可视化
            if vis_count < VIS_NUM:
                for i in range(len(file_names)):
                    if vis_count >= VIS_NUM:
                        break
                    vis_save_path = os.path.join(vis_dir, f"{file_names[i]}.png")
                    vis_segvitrs_result(
                        imgs[i], masks[i], preds[i],
                        file_names[i], vis_save_path,
                        train_mean, train_std
                    )
                    # 保存预测图
                    pred = preds[i].cpu().numpy()
                    pred_color = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    for cls in range(NUM_CLASSES):
                        pred_color[pred == cls] = CLASS_COLORS[cls]
                    pred_save_path = os.path.join(pred_map_dir, f"{file_names[i]}_pred.png")
                    cv2.imwrite(pred_save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                    vis_count += 1

    # 计算并保存指标
    results = metrics.compute()
    print("\n" + "=" * 60)
    print("📊 SegViTRS_Improved 模型测试结果汇总")
    print("=" * 60)
    print(f"数据集: {DATASET_NAME}")
    print(f"总体精度（OA）: {results['OA']:.4f}")
    print(f"平均IoU（mIoU）: {results['mIoU']:.4f}")
    print(f"平均精确率: {results['mPrecision']:.4f}")
    print(f"平均召回率: {results['mRecall']:.4f}")
    print(f"平均F1分数: {results['mF1']:.4f}")

    # 打印每类指标
    print("\n📋 每类详细指标:")
    for cls_name in CLASS_NAMES:
        print(f"\n{cls_name}:")
        print(f"  IoU: {results['per_class_iou'][cls_name]:.4f}")
        print(f"  精确率: {results['per_class_precision'][cls_name]:.4f}")
        print(f"  召回率: {results['per_class_recall'][cls_name]:.4f}")
        print(f"  F1: {results['per_class_f1'][cls_name]:.4f}")

    # 保存指标到JSON
    metrics_path = os.path.join(test_output_dir, 'segvitrs_test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 生成测试报告
    report = f"""
# SegViTRS_Improved 模型测试报告
## 测试配置
- 数据集: {DATASET_NAME}
- 样本数量: {len(test_dataset)}
- 输入尺寸: {IMG_SIZE}×{IMG_SIZE}
- Batch Size: {BATCH_SIZE}
- 输入通道数: {IN_CHANNELS}
- 模型权重: {os.path.basename(MODEL_WEIGHT_PATH)}
- 归一化均值: {train_mean.round(4).tolist()}
- 归一化标准差: {train_std.round(4).tolist()}

## 核心指标
| 指标 | 数值 |
|------|------|
| OA (总体精度) | {results['OA']:.4f} |
| mIoU (平均IoU) | {results['mIoU']:.4f} |
| 平均精确率 | {results['mPrecision']:.4f} |
| 平均召回率 | {results['mRecall']:.4f} |
| 平均F1分数 | {results['mF1']:.4f} |

## 每类IoU
"""
    for i, cls_name in enumerate(CLASS_NAMES):
        report += f"- {cls_name}: {results['per_class_iou'][cls_name]:.4f}\n"

    report_path = os.path.join(test_output_dir, 'segvitrs_test_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ SegViTRS_Improved 测试完成！")
    print(f"📁 指标文件: {metrics_path}")
    print(f"📄 报告文件: {report_path}")
    print(f"🎨 可视化结果: {vis_dir}")
    print(f"🗺️ 预测图: {pred_map_dir}")

# ======================== 模型直接写在文件内，无任何外部导入 ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple

# --------------------------- 基础配置 ---------------------------
CFG = {
    "in_channels": 3,
    "num_classes": 6,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "patch_size": 4,
    "window_size": 8,
    "mlp_ratio": 4.,
    "drop_rate": 0.1,
    "use_geo_pos_encoding": True,
    "use_land_prior": True,
    "decoder_embed_dim": 64
}

# --------------------------- 基础模块 ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class GeoSpatialPosEncoding(nn.Module):
    def __init__(self, embed_dim: int, img_size: int = 256, patch_size: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        x_pos = torch.linspace(-1, 1, img_size // patch_size)
        y_pos = torch.linspace(-1, 1, img_size // patch_size)
        y, x = torch.meshgrid(y_pos, x_pos, indexing="ij")
        self.x_embed = nn.Linear(1, embed_dim // 2)
        self.y_embed = nn.Linear(1, embed_dim // 2)
        self.register_buffer('x_coord', x.reshape(-1, 1))
        self.register_buffer('y_coord', y.reshape(-1, 1))

    def forward(self) -> torch.Tensor:
        x_embed = self.x_embed(self.x_coord)
        y_embed = self.y_embed(self.y_coord)
        pos_embed = torch.cat([x_embed, y_embed], dim=-1)
        return pos_embed.unsqueeze(0)

class LandPriorFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.land_weights = nn.Parameter(torch.ones(6, in_channels))
        self.conv = ConvBNReLU(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        weights = self.softmax(self.land_weights).mean(dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        weighted_feat = x * weights
        return self.conv(weighted_feat)

class CrossLayerFeatureFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_channel = x * self.channel_att(x)
        avg = torch.mean(x_channel, dim=1, keepdim=True)
        max, _ = torch.max(x_channel, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max], dim=1)
        x_spatial = x_channel * self.spatial_att(x_cat)
        return x_spatial

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, drop_rate: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q / (self.head_dim ** 0.5)
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop_rate: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SegViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4., drop_rate: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, window_size, drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop_rate=drop_rate)
        self.cnn_feat = ConvBNReLU(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x_2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        cnn_feat = self.cnn_feat(x_2d)
        cnn_feat = rearrange(cnn_feat, 'b c h w -> b (h w) c')
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x + cnn_feat
        return x

class SegViTEncoder(nn.Module):
    def __init__(self, cfg: dict, img_size: int = 256):
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg["embed_dim"]
        depths = cfg["depths"]
        num_heads = cfg["num_heads"]
        patch_size = cfg["patch_size"]
        window_size = cfg["window_size"]
        mlp_ratio = cfg["mlp_ratio"]
        drop_rate = cfg["drop_rate"]

        self.patch_embed = nn.Conv2d(cfg["in_channels"], embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.h, self.w = img_size // patch_size, img_size // patch_size

        if cfg["use_geo_pos_encoding"]:
            self.pos_encoding = GeoSpatialPosEncoding(embed_dim, img_size, patch_size)

        self.stage_channels = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.feature_fusions = nn.ModuleList()

        for i in range(len(depths)):
            stage_blocks = nn.ModuleList([
                SegViTBlock(self.stage_channels[i], num_heads[i], window_size, mlp_ratio, drop_rate)
                for _ in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            self.feature_fusions.append(CrossLayerFeatureFusion(self.stage_channels[i]))
            if i < len(depths) - 1:
                downsample = nn.Conv2d(self.stage_channels[i], self.stage_channels[i+1], 2, 2)
                self.downsamples.append(downsample)

        if cfg["use_land_prior"]:
            self.land_fusion = LandPriorFusion(self.stage_channels[-1], self.stage_channels[-1])

    def forward(self, x: torch.Tensor):
        features = []
        hs, ws = [], []
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        if self.cfg["use_geo_pos_encoding"]:
            x = x + self.pos_encoding()

        for i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x, h, w)
            x_2d = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x_2d = self.feature_fusions[i](x_2d)
            features.append(x_2d)
            hs.append(h)
            ws.append(w)
            if i < len(self.downsamples):
                x_2d = self.downsamples[i](x_2d)
                b, c, h, w = x_2d.shape
                x = rearrange(x_2d, 'b c h w -> b (h w) c')

        if self.cfg["use_land_prior"]:
            features[-1] = self.land_fusion(features[-1])

        return features, hs, ws, self.stage_channels

class SwinUpsample(nn.Module):
    def __init__(self, dim, scale_factor=2):
        super().__init__()
        self.scale = scale_factor
        self.proj = nn.Linear(dim, dim * (scale_factor ** 2))

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'B C H W -> B (H W) C')
        x = self.proj(x)
        x = rearrange(x, 'B (H W) (s1 s2 C) -> B C (H s1) (W s2)',
                      H=H, W=W, s1=self.scale, s2=self.scale)
        return x

class DecoderTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, window_size=8)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))

    def forward(self, x, h, w):
        B, L, C = x.shape
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SegViTDecoder(nn.Module):
    def __init__(self, cfg: dict, stage_channels: List[int]):
        super().__init__()
        self.stage_channels = stage_channels
        dec_dim = cfg["decoder_embed_dim"]
        self.num_stages = len(stage_channels)

        self.upsamples = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        current_dim = stage_channels[-1]
        for i in range(self.num_stages-1, 0, -1):
            self.upsamples.append(SwinUpsample(current_dim))
            fuse_dim = current_dim + stage_channels[i-1]
            self.fusion_blocks.append(nn.Linear(fuse_dim, dec_dim))
            self.decoder_blocks.append(DecoderTransformerBlock(dec_dim))
            current_dim = dec_dim

    def forward(self, features: List[torch.Tensor], hs: List[int], ws: List[int]):
        x = features[-1]
        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            skip = features[-(i+2)]
            x = F.interpolate(x, size=skip.shape[2:])
            x = rearrange(x, 'B C H W -> B (H W) C')
            skip = rearrange(skip, 'B C H W -> B (H W) C')
            x = torch.cat([x, skip], dim=-1)
            x = self.fusion_blocks[i](x)
            h, w = hs[-(i+2)], ws[-(i+2)]
            x = self.decoder_blocks[i](x, h, w)
            x = rearrange(x, 'B (H W) C -> B C H W', H=h, W=w)
        return x

class EnhancedSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, patch_size):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear')
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.pwconv = nn.Conv2d(embed_dim, embed_dim//2, 1)
        self.norm = nn.BatchNorm2d(embed_dim//2)
        self.act = nn.GELU()
        self.out = nn.Conv2d(embed_dim//2, num_classes, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.act(self.norm(x))
        return self.out(x)

class SegViTRS_Improved(nn.Module):
    def __init__(self, cfg=CFG, img_size=256):
        super().__init__()
        self.encoder = SegViTEncoder(cfg, img_size)
        self.decoder = SegViTDecoder(cfg, self.encoder.stage_channels)
        self.seg_head = EnhancedSegHead(cfg["decoder_embed_dim"], cfg["num_classes"], cfg["patch_size"])

    def forward(self, x):
        features, hs, ws, _ = self.encoder(x)
        feat = self.decoder(features, hs, ws)
        out = self.seg_head(feat)
        return out

if __name__ == "__main__":
    main()