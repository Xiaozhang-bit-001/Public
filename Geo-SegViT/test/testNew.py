import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from datetime import datetime
import cv2

# ====================== 修复 matplotlib 报错 ======================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================== 修复权重加载安全 ======================
import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([
    argparse.Namespace,
    numpy.core.multiarray.scalar
])

# ====================== 模型导入 ======================
from QingModel import SegViTRS_Improved

# ====================== 类别颜色与名称 ======================
CLASS_NAMES = [
    'Low shrub', 'Impervious surfaces', 'Building',
    'Background', 'Vegetation', 'Vehicle'
]

LABEL_TO_RGB = {
    0: (0, 255, 255),
    1: (255, 255, 255),
    2: (0, 0, 255),
    3: (255, 0, 0),
    4: (0, 255, 0),
    5: (255, 255, 0)
}

def label_to_rgb(label_array):
    h, w = label_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in LABEL_TO_RGB.items():
        rgb[label_array == cls] = color
    return rgb

# ====================== 数据集 ======================
class RemoteSensingDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, list_path, img_size=256, mean=None, std=None):
        self.root_path = root_path
        self.img_size = img_size
        self.mean = np.array(mean)
        self.std = np.array(std)

        with open(list_path, 'r') as f:
            self.file_list = [x.strip() for x in f.readlines()]
        self.file_list = [x.replace('.npz', '') for x in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        data = np.load(os.path.join(self.root_path, f"{fname}.npz"))
        img = data['image']
        label = data['label']

        # (3,H,W) -> 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean.reshape(3,1,1)) / self.std.reshape(3,1,1)

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).long()
        return img_tensor, label_tensor, fname

# ====================== 指标计算（含 OA） ======================
def compute_metrics(preds, labels, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    for p, l in zip(preds, labels):
        p = p.flatten()
        l = l.flatten()
        valid = l < num_classes
        p = p[valid]
        l = l[valid]

        total_correct += (p == l).sum()
        total_pixels += len(l)

        for pi, li in zip(p, l):
            confusion[li, pi] += 1

    iou, prec, rec, f1 = [], [], [], []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        iou.append(tp / (tp + fp + fn + 1e-8))
        prec.append(tp / (tp + fp + 1e-8))
        rec.append(tp / (tp + fn + 1e-8))
        f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1] + 1e-8))

    oa = total_correct / (total_pixels + 1e-8)

    return {
        "IoU": [round(x, 4) for x in iou],
        "Precision": [round(x, 4) for x in prec],
        "Recall": [round(x, 4) for x in rec],
        "F1": [round(x, 4) for x in f1],
        "OA": round(oa, 4),
        "mIoU": round(np.mean(iou), 4),
        "mPrec": round(np.mean(prec), 4),
        "mRec": round(np.mean(rec), 4),
        "mF1": round(np.mean(f1), 4)
    }

# ====================== 主函数 ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SegViTRS_Improved')
    parser.add_argument('--model_path',
                        default=r'G:\erya308\zhangjunming\modeL\Geo-SegViT\ComparedModel2_Best\SegViTRS_Improved_Pot_networks\SegViTRS_Improved_Pots_256_256\iter30k_epo150_bs8_lr0.01_s1234\best_model.pth')
    parser.add_argument('--root_path',
                        default=r'G:\erya308\zhangjunming\modeL\Geo-SegViT\datasets\Potsdam\npz_data_RGB_improved')
    parser.add_argument('--list_path',
                        default=r'G:\erya308\zhangjunming\modeL\Geo-SegViT\datasets\Potsdam\lists_txt_RGB_improved\test.txt')
    parser.add_argument('--data_stats_path',
                        default=r'G:\erya308\zhangjunming\modeL\Geo-SegViT\datasets\Potsdam\rgb_data_stats_improved.npz')
    parser.add_argument('--num_classes', default=6)
    parser.add_argument('--img_size', default=256)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--vis_save_dir', default=r'G:\erya308\zhangjunming\modeL\Geo-SegViT\test_results_final_MP')
    args = parser.parse_args()

    device = torch.device('cuda')

    # 加载均值方差
    stats = np.load(args.data_stats_path)
    mean = stats["mean"] / 255.0
    std = stats["std"] / 255.0

    # 数据集
    dataset = RemoteSensingDataset(
        args.root_path, args.list_path,
        mean=mean, std=std
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    cfg = {
        "in_channels": 3, "num_classes": 6, "embed_dim": 96, "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24], "patch_size": 4, "window_size": 8,
        "mlp_ratio": 4.0, "drop_rate": 0.1, "use_geo_pos_encoding": True,
        "use_land_prior": True, "decoder_embed_dim": 64
    }
    model = SegViTRS_Improved(cfg=cfg, img_size=256).to(device)

    # 加载权重
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    # 推理
    preds_all, gts_all, imgs_all, fnames_all = [], [], [], []
    print("\n🚀 开始推理...")

    with torch.no_grad():
        for imgs, labels, fnames in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1).cpu().numpy()

            imgs_all.extend(imgs.cpu().numpy())
            preds_all.append(pred)
            gts_all.append(labels.numpy())
            fnames_all.extend(fnames)

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)

    # 计算指标
    metrics = compute_metrics(preds_all, gts_all, args.num_classes)

    # ====================== 输出日志 ======================
    print("\n" + "=" * 80)
    print("📊 测试结果指标")
    print("=" * 80)
    print(f"OA (总体精度):      {metrics['OA']:.4f}")
    print(f"mIoU (平均交并比):  {metrics['mIoU']:.4f}")
    print(f"mPrec (精确率):     {metrics['mPrec']:.4f}")
    print(f"mRecall (召回率):   {metrics['mRec']:.4f}")
    print(f"mF1 (平均F1):       {metrics['mF1']:.4f}")
    print("\n📋 每类指标：")
    for c in range(args.num_classes):
        print(f"类别{c} {CLASS_NAMES[c]:15s} | IoU={metrics['IoU'][c]:.4f}  Prec={metrics['Precision'][c]:.4f}  Recall={metrics['Recall'][c]:.4f}  F1={metrics['F1'][c]:.4f}")
    print("=" * 80)

    # ====================== 创建文件夹 ======================
    save_vis = os.path.join(args.vis_save_dir, "vis_three_panel")    # 三分图
    save_pred = os.path.join(args.vis_save_dir, "predictions_only") # 纯预测图
    save_yuan = os.path.join(args.vis_save_dir, "yuanshi")  #原始图像
    os.makedirs(save_vis, exist_ok=True)
    os.makedirs(save_pred, exist_ok=True)
    os.makedirs(save_yuan,exist_ok=True)

    # ====================== 保存图像 ======================
    print("\n📸 正在保存图像...")
    for i in range(len(preds_all)):
        fname = fnames_all[i]
        img = imgs_all[i]
        pred = preds_all[i]
        gt = gts_all[i]

        # 原图还原
        img_vis = img.transpose(1,2,0)
        img_vis = img_vis * std + mean
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)

        # 彩色图
        gt_rgb = label_to_rgb(gt)
        pred_rgb = label_to_rgb(pred)

        # 保存：纯预测图
        cv2.imwrite(os.path.join(save_pred, f"{fname}.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

        # 保存原始图像
        cv2.imwrite(os.path.join(save_yuan, f"{fname}.png"),cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

        # 保存：三分图（原图 + GT + 预测）
        plt.figure(figsize=(15, 5))
        plt.subplot(1,3,1); plt.imshow(img_vis); plt.title("Image"); plt.axis('off')
        plt.subplot(1,3,2); plt.imshow(gt_rgb); plt.title("Ground Truth"); plt.axis('off')
        plt.subplot(1,3,3); plt.imshow(pred_rgb); plt.title("Prediction"); plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_vis, f"{fname}.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 保存指标日志
    log_path = os.path.join(args.vis_save_dir, "test_metrics.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 三分图保存至：{save_vis}")
    print(f"✅ 预测图保存至：{save_pred}")
    print(f"✅ 指标日志保存至：{log_path}")
    print("🎉 测试全部完成！")

if __name__ == "__main__":
    main()