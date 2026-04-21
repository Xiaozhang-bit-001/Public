import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys

# 解决PyTorch 2.6+权重加载安全机制
import torch.serialization

torch.serialization.add_safe_globals([object])

# 添加项目根目录到Python路径（解决model2导入问题）
sys.path.append(r"G:\erya308\zhangjunming\ST-UNet")

# 设置中文显示（可视化用）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# -------------------------- 核心标签映射配置 --------------------------
RGB_LABEL_MAPPING = {
    (0, 255, 255): 0,  # 低灌木（青色）
    (255, 255, 255): 1,  # 不透水面（白色）
    (0, 0, 255): 2,  # 建筑（纯蓝色）
    (255, 0, 0): 3,  # 背景（纯红色）
    (0, 255, 0): 4,  # 植被（纯绿色）
    (255, 255, 0): 5  # 车辆（纯黄色）
}

# 反向映射：类别索引转RGB颜色
LABEL_RGB_MAPPING = {v: k for k, v in RGB_LABEL_MAPPING.items()}

# 类别名称（按索引对应）
CLASS_NAMES = [
    'Low shrub',  # 0 - 低灌木
    'Impervious surfaces',  # 1 - 不透水面
    'Building',  # 2 - 建筑
    'Background',  # 3 - 背景
    'Vegetation',  # 4 - 植被
    'Vehicle'  # 5 - 车辆
]

# 测试集实际存在的类别（全6类）
VALID_CLASSES = [0, 1, 2, 3, 4, 5]
# 类别配色（按索引对应）
CLASS_COLORS = [
    (0, 255, 255),  # 0 - 青色-低灌木
    (255, 255, 255),  # 1 - 白色-不透水面
    (0, 0, 255),  # 2 - 蓝色-建筑
    (255, 0, 0),  # 3 - 红色-背景
    (0, 255, 0),  # 4 - 绿色-植被
    (255, 255, 0)  # 5 - 黄色-车辆
]


# -------------------------- 固定参数配置（DANet） --------------------------
class Args:
    def __init__(self):
        # 数据配置
        self.root_path = r"G:\erya308\zhangjunming\Geo-SegViT\datasets\Vaihingen\npz_data_RGB_improved"
        self.list_dir = r"G:\erya308\zhangjunming\Geo-SegViT\datasets\Vaihingen\lists_txt_RGB_improved"
        self.img_size = 256
        self.batch_size = 8
        self.num_classes = 6

        # 模型配置（DANet）
        self.model_name = "DANet"
        self.model_weight_path = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\FinalResults\DANet\Vai_256\DANet_Vai_256_256_bs8_ep100_final_20251205_192528.pth"
        self.config_path = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\FinalResults\DANet\Vai_256\DANet_Vai_256_256_bs8_ep100_config_20251205_192528.json"

        # 输出配置
        self.output_dir = r"G:\erya308\zhangjunming\Geo-SegViT\ComparedModels\TestResults"
        self.vis_num = 10
        self.save_pred_maps = True


args = Args()


# -------------------------- 数据加载 --------------------------
class VaihingenDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, list_dir, split='test', img_size=256, transform=None):
        self.root_path = root_path
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # 扫描目录加载所有npz文件
        self.file_list = []
        for f in os.listdir(root_path):
            if f.endswith('.npz'):
                self.file_list.append(os.path.splitext(f)[0])
        print(f"📁 扫描模式：从目录加载{len(self.file_list)}个npz文件")

        # 过滤有效文件
        self.valid_files = []
        self.missing_files = []
        for file_name in self.file_list:
            npz_path = os.path.join(root_path, f"{file_name}.npz")
            if os.path.exists(npz_path):
                self.valid_files.append(file_name)
            else:
                self.missing_files.append(file_name)

        self.file_list = self.valid_files
        print(f"✅ 过滤后有效样本数：{len(self.valid_files)}")
        if self.missing_files:
            print(f"⚠️ 缺失文件数：{len(self.missing_files)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        npz_path = os.path.join(self.root_path, f"{file_name}.npz")

        # 加载数据
        if not os.path.exists(npz_path):
            print(f"警告：文件不存在 {npz_path}，返回空数据")
            img = np.zeros((256, 256, 3), dtype=np.float32)
            mask = np.zeros((256, 256), dtype=np.int64)
        else:
            data = np.load(npz_path)
            img = data['image']  # (3,256,256)
            mask = data['label']  # (256,256)
            # 调整图像维度为 (H,W,3)
            if len(img.shape) == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

        # 归一化
        if self.transform is not None:
            img = self.transform(img)

        # 转换为tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name


# -------------------------- 评价指标计算 --------------------------
class SegmentationMetrics:
    def __init__(self, num_classes, valid_classes):
        self.num_classes = num_classes
        self.valid_classes = valid_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    def update(self, pred, target):
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        mask = (target >= 0) & (target < self.num_classes) & (np.isin(target, self.valid_classes))
        pred = pred[mask]
        target = target[mask]
        if len(pred) > 0 and len(target) > 0:
            self.confusion_matrix += confusion_matrix(target, pred, labels=range(self.num_classes))

    def compute(self):
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = intersection / (union + 1e-6)
        precision = intersection / (cm.sum(axis=0) + 1e-6)
        recall = intersection / (cm.sum(axis=1) + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        valid_mask = np.isin(range(self.num_classes), self.valid_classes)
        oa = np.diag(cm)[valid_mask].sum() / (cm[valid_mask, :][:, valid_mask].sum() + 1e-6)
        miou = np.mean(iou[valid_mask])
        mprecision = np.mean(precision[valid_mask])
        mrecall = np.mean(recall[valid_mask])
        mf1 = np.mean(f1[valid_mask])
        return {
            'per_class_iou': {CLASS_NAMES[i]: float(iou[i]) for i in range(self.num_classes)},
            'per_class_precision': {CLASS_NAMES[i]: float(precision[i]) for i in range(self.num_classes)},
            'per_class_recall': {CLASS_NAMES[i]: float(recall[i]) for i in range(self.num_classes)},
            'per_class_f1': {CLASS_NAMES[i]: float(f1[i]) for i in range(self.num_classes)},
            'mIoU': float(miou),
            'OA': float(oa),
            'mPrecision': float(mprecision),
            'mRecall': float(mrecall),
            'mF1': float(mf1),
            'valid_classes': self.valid_classes
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)


# -------------------------- 可视化函数 --------------------------
def vis_segmentation(img, mask, pred, file_name, save_path, mean, std):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    def mask_to_color(mask_data, colors):
        h, w = mask_data.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(colors)):
            color_img[mask_data == i] = colors[i]
        return color_img

    mask_color = mask_to_color(mask, CLASS_COLORS)
    pred_color = mask_to_color(pred, CLASS_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('原图')
    axes[0].axis('off')
    axes[1].imshow(mask_color)
    axes[1].set_title('真实标签')
    axes[1].axis('off')
    axes[2].imshow(pred_color)
    axes[2].set_title('预测结果')
    axes[2].axis('off')
    plt.suptitle(file_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# -------------------------- 主测试函数 --------------------------
def main():
    # 1. 创建输出目录
    test_output_dir = os.path.join(args.output_dir, args.model_name)
    vis_dir = os.path.join(test_output_dir, 'visualization')
    pred_map_dir = os.path.join(test_output_dir, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)

    # 2. 加载训练配置
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            train_config = json.load(f)
    except UnicodeDecodeError:
        with open(args.config_path, 'r', encoding='gbk') as f:
            train_config = json.load(f)

    # 3. 构建归一化函数
    train_mean = np.array(train_config['dataset_stats']['mean'])
    train_std = np.array(train_config['dataset_stats']['std'])

    def transform(img):
        img = img.astype(np.float32)
        mean = train_mean.reshape(1, 1, 3)
        std = train_std.reshape(1, 1, 3)
        img = (img - mean) / std
        return img

    # 4. 加载数据集
    test_dataset = VaihingenDataset(
        root_path=args.root_path,
        list_dir=args.list_dir,
        split='test',
        img_size=args.img_size,
        transform=transform
    )
    actual_npz_files = [f for f in os.listdir(args.root_path) if f.endswith('.npz')]
    print(f"\n📊 测试集完整统计：")
    print(f"目录下所有npz文件数: {len(actual_npz_files)}")
    print(f"加载的有效样本数: {len(test_dataset)}")
    print(f"缺失文件数: {len(test_dataset.missing_files)}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"✅ 加载测试集完成，共{len(test_dataset)}个有效样本")

    # 5. 初始化DANet模型
    print(f"\n📌 初始化模型: {args.model_name}")
    from model2.DANet import DANet
    model = DANet(
        num_classes=args.num_classes,
        in_channels=3
    ).to(device)

    # 6. 加载模型权重
    checkpoint = torch.load(args.model_weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print(f"✅ 加载模型权重完成: {args.model_weight_path}")

    # 7. 初始化评价指标
    metrics = SegmentationMetrics(args.num_classes, VALID_CLASSES)

    # 8. 批量测试
    print("\n🚀 开始测试...")
    vis_count = 0
    with torch.no_grad():
        for imgs, masks, file_names in tqdm(test_loader, desc='测试进度'):
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            metrics.update(preds, masks)
            # 保存预测图和可视化
            for i in range(len(file_names)):
                if args.save_pred_maps:
                    pred = preds[i].cpu().numpy()
                    pred_color = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
                    for c in range(args.num_classes):
                        pred_color[pred == c] = CLASS_COLORS[c]
                    save_path = os.path.join(pred_map_dir, f"{file_names[i]}_pred.png")
                    cv2.imwrite(save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                if vis_count < args.vis_num:
                    vis_save_path = os.path.join(vis_dir, f"{file_names[i]}_vis.png")
                    vis_segmentation(imgs[i], masks[i], preds[i], file_names[i], vis_save_path, train_mean, train_std)
                    vis_count += 1

    # 9. 计算并保存结果
    results = metrics.compute()
    print("\n" + "=" * 60)
    print("📊 测试结果汇总（全6类评估）")
    print("=" * 60)
    print(f"总体精度（OA）: {results['OA']:.4f}")
    print(f"平均IoU（mIoU）: {results['mIoU']:.4f}")
    print(f"平均精确率（mPrecision）: {results['mPrecision']:.4f}")
    print(f"平均召回率（mRecall）: {results['mRecall']:.4f}")
    print(f"平均F1分数（mF1）: {results['mF1']:.4f}")
    print("\n📋 每类详细指标:")
    for cls_name in CLASS_NAMES:
        print(f"\n{cls_name}:")
        print(f"  IoU: {results['per_class_iou'][cls_name]:.4f}")
        print(f"  Precision: {results['per_class_precision'][cls_name]:.4f}")
        print(f"  Recall: {results['per_class_recall'][cls_name]:.4f}")
        print(f"  F1: {results['per_class_f1'][cls_name]:.4f}")
    # 保存指标
    metrics_path = os.path.join(test_output_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 测试指标已保存到: {metrics_path}")
    # 生成报告
    report = f"""# {args.model_name} 测试报告
## 测试配置
- 数据集: Vaihingen (test集)
- 样本数量: {len(test_dataset)}
- 输入尺寸: {args.img_size}×{args.img_size}
- 批次大小: {args.batch_size}
- 模型权重: {os.path.basename(args.model_weight_path)}

## 核心指标（全6类评估）
| 指标 | 数值 |
|------|------|
| OA (总体精度) | {results['OA']:.4f} |
| mIoU (平均IoU) | {results['mIoU']:.4f} |
| mPrecision (平均精确率) | {results['mPrecision']:.4f} |
| mRecall (平均召回率) | {results['mRecall']:.4f} |
| mF1 (平均F1) | {results['mF1']:.4f} |

## 每类IoU
"""
    for i, cls_name in enumerate(CLASS_NAMES):
        report += f"- {cls_name}: {results['per_class_iou'][cls_name]:.4f}\n"
    report_path = os.path.join(test_output_dir, 'test_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 测试报告已保存到: {report_path}")
    print(f"\n🎉 测试完成！所有结果已保存到: {test_output_dir}")


if __name__ == "__main__":
    main()