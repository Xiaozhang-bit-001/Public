import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
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

# ======================== 1. 核心配置 ========================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据集配置
DATASET_NAME = 'Pots_256'
ROOT_PATH = r'/root/autodl-tmp/Geo-SegViT/datasets/Potsdam/npz_data_RGB_improved'
DATA_STATS_PATH = r'/root/autodl-tmp/Geo-SegViT/datasets/Potsdam/rgb_data_stats_improved.npz'

# 模型配置
MODEL_NAME = 'SwinUNet'
MODEL_WEIGHT_PATH = r"/root/autodl-tmp/Geo-SegViT/ComparedModels_U/SwinUNet_P/best.pth"

# 超参数
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 6
IN_CHANNELS = 3
VIS_NUM = 100
DEBUG = True

# SwinUNet 参数
EMBED_DIM = 96
DEPTHS = [2, 2, 6, 2]
NUM_HEADS = [3, 6, 12, 24]

# 输出配置
OUTPUT_DIR = r'/root/autodl-tmp/Geo-SegViT/TestResults/SwinUNet_P'

# 类别配置
CLASS_NAMES = [
    'Low shrub', 'Impervious surfaces', 'Building',
    'Background', 'Vegetation', 'Vehicle'
]
VALID_CLASSES = [0, 1, 2, 3, 4, 5]
CLASS_COLORS = [
    (0, 255, 255),
    (255, 255, 255),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0)
]


# ======================== 2. 数据加载 ========================
class SwinUNetTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, img_size=256, transform=None):
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform

        self.file_paths = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_paths.append(os.path.join(root_path, f))

        print(f"📁 加载{len(self.file_paths)}个测试样本")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        file_name = os.path.basename(npz_path).replace('.npz', '')

        data = np.load(npz_path)
        img = data['image']
        mask = data['label']

        assert img.shape == (3, self.img_size, self.img_size), f"图像维度错误: {img.shape}"
        assert mask.shape == (self.img_size, self.img_size), f"标签维度错误: {mask.shape}"

        if self.transform is not None:
            img_trans = img.transpose(1, 2, 0).astype(np.float32)
            img_trans = self.transform(img_trans)
            img = img_trans.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        if DEBUG and idx == 0:
            unique_labels = np.unique(mask.numpy())
            print(f"🔍 样本{file_name}标签分布: {unique_labels}")

        return img, mask, file_name


# ======================== 3. 评价指标 ========================
class SwinUNetMetrics:
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

        valid_mask = np.array([cls in VALID_CLASSES for cls in range(self.num_classes)])
        results['mIoU'] = float(np.mean(np.array(iou)[valid_mask]))
        results['mPrecision'] = float(np.mean(np.array(precision)[valid_mask]))
        results['mRecall'] = float(np.mean(np.array(recall)[valid_mask]))
        results['mF1'] = float(np.mean(np.array(f1)[valid_mask]))

        total_tp = np.diag(cm).sum()
        total_samples = cm.sum()
        results['OA'] = float(total_tp / (total_samples + 1e-8))

        results['per_class_iou'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(iou)}
        results['per_class_precision'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(precision)}
        results['per_class_recall'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(recall)}
        results['per_class_f1'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(f1)}

        return results


# ======================== 4. 可视化函数（与FCN完全统一） ========================
def vis_swinunet_result(img, mask, pred, file_name, save_path, mean, std):
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * std + mean)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    def mask2color(mask_data):
        h, w = mask_data.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in range(len(CLASS_COLORS)):
            color_img[mask_data == cls] = CLASS_COLORS[cls]
        return color_img

    mask_color = mask2color(mask.cpu().numpy())
    pred_color = mask2color(pred.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Label', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Pred', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================== 6. 主函数 ========================
def main():
    PROJECT_ROOT = r'/root/autodl-tmp/Geo-SegViT'
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    test_output_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    vis_dir = os.path.join(test_output_dir, 'visualization')
    pred_map_dir = os.path.join(test_output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)
    print(f"📌 测试结果保存路径: {test_output_dir}")

    stats = np.load(DATA_STATS_PATH)
    train_mean = stats['mean']
    train_std = stats['std']
    print(f"📊 均值: {train_mean.round(4)}")
    print(f"📊 标准差: {train_std.round(4)}")

    def transform(img):
        return (img - train_mean) / train_std

    test_dataset = SwinUNetTestDataset(ROOT_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 加载模型
    print("\n🔧 加载 SwinUNet 模型...")
    from model2.SwinUnet import SwinUNet

    model = SwinUNet(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    # 加载权重（与FCN完全一致）
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 权重严格加载成功")
    except:
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ 权重非严格加载成功")

    model.eval()
    print(f"✅ 模型加载完成")

    metrics = SwinUNetMetrics(NUM_CLASSES)
    vis_count = 0

    with torch.no_grad():
        for imgs, masks, file_names in tqdm(test_loader, desc="测试中"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

            if vis_count < VIS_NUM:
                for i in range(len(file_names)):
                    vis_swinunet_result(
                        imgs[i], masks[i], preds[i],
                        file_names[i],
                        os.path.join(vis_dir, f"{file_names[i]}.png"),
                        train_mean, train_std
                    )
                    vis_count += 1

    results = metrics.compute()
    print("\n" + "=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    print(f"OA: {results['OA']:.4f}")
    print(f"mIoU: {results['mIoU']:.4f}")
    print(f"mF1: {results['mF1']:.4f}")

    with open(os.path.join(test_output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()