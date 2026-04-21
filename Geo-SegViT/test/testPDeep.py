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
MODEL_WEIGHT_PATH = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\FinalResultsP\DeeplabV3Plus\Pots_256\DeeplabV3Plus_Pots_256_256_bs8_ep100_final_20251209_114335.pth"
CONFIG_PATH = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\FinalResultsP\DeeplabV3Plus\Pots_256\DeeplabV3Plus_Pots_256_256_bs8_ep100_config_20251209_114335.json"

# 超参数（与模型/训练一致）
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 6
IN_CHANNELS = 3
VIS_NUM = 10
DEBUG = True

# 输出配置
OUTPUT_DIR = r'G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\TestResults\DeeplabV3Plus_P'

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


# ======================== 2. 数据加载（修复所有已知错误） ========================
class DeeplabV3PlusTestDataset(torch.utils.data.Dataset):
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
            unique_labels = np.unique(mask)  # 修复：移除.numpy()
            print(f"\n🔍 样本{file_name}标签分布: {unique_labels}")
            print(f"🔍 输入图像像素范围: [{img.min():.2f}, {img.max():.2f}]")
            print(f"🔍 标签值范围: [{mask.min()}, {mask.max()}]")

        # 转换为tensor
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name


# ======================== 3. 评价指标计算 ========================
class DeeplabV3PlusMetrics:
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

        # 每类详细指标
        results['per_class_iou'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(iou)}
        results['per_class_precision'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(precision)}
        results['per_class_recall'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(recall)}
        results['per_class_f1'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(f1)}

        # 调试：输出混淆矩阵
        if DEBUG:
            print("\n🔍 混淆矩阵:")
            print(cm)

        return results


# ======================== 4. 可视化函数（适配模型输出） ========================
def vis_deeplabv3plus_result(img, mask, pred, file_name, save_path, mean, std):
    # 反归一化（适配均值类型）
    img_np = img.cpu().numpy().transpose(1, 2, 0)  # (3,H,W) → (H,W,3)
    img_np = img_np * std + mean

    # 根据均值范围调整反归一化
    if mean.max() > 1:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # 标签转彩色图（兼容Tensor/NumPy）
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
    axes[2].set_title('DeeplabV3Plus预测结果', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(file_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================== 5. 主函数（核心适配模型初始化） ========================
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

    # 数据预处理（适配均值类型）
    def transform(img):
        img = img.astype(np.float32)
        # 若均值是0-255范围，无需/255
        if train_mean.max() > 1:
            img = (img - train_mean) / train_std
        else:
            img = img / 255.0
            img = (img - train_mean) / train_std
        return img

    # 加载数据集
    test_dataset = DeeplabV3PlusTestDataset(
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

    # 加载模型（完全匹配你的模型定义）
    print("\n🔧 加载DeeplabV3Plus模型...")
    from model2.DeepLabVp import DeeplabV3Plus  # 导入你的模型类
    # 仅传入模型支持的参数：num_classes + in_channels
    model = DeeplabV3Plus(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS
    ).to(DEVICE)

    # 加载权重
    try:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)

    # 处理多GPU权重前缀
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 验证权重加载
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f"✅ 权重加载成功（strict=True）")
    except RuntimeError as e:
        print(f"⚠️ 权重严格加载失败: {e}")
        print("🔄 尝试非严格加载...")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # 验证模型输出有效性
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
    metrics = DeeplabV3PlusMetrics(NUM_CLASSES)
    metrics.reset()

    # 开始测试
    print("\n🚀 开始测试DeeplabV3Plus模型...")
    vis_count = 0
    with torch.no_grad():
        for batch_idx, (imgs, masks, file_names) in enumerate(tqdm(test_loader, desc='DeeplabV3Plus测试进度')):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 前向推理（模型返回单输出）
            outputs = model(imgs)  # (B, 6, 256, 256)
            preds = torch.argmax(outputs, dim=1)  # (B, 256, 256)

            # 调试：第一个batch预测分布
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
                    vis_deeplabv3plus_result(
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
    print("📊 DeeplabV3Plus模型测试结果汇总")
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
    metrics_path = os.path.join(test_output_dir, 'deeplabv3plus_test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 生成测试报告
    report = f"""
# DeeplabV3Plus模型测试报告
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

    report_path = os.path.join(test_output_dir, 'deeplabv3plus_test_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ DeeplabV3Plus测试完成！")
    print(f"📁 指标文件: {metrics_path}")
    print(f"📄 报告文件: {report_path}")
    print(f"🎨 可视化结果: {vis_dir}")
    print(f"🗺️ 预测图: {pred_map_dir}")


if __name__ == "__main__":
    main()