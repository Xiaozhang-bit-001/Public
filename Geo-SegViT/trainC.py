import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入其他对比模型
from model2.HRViT_RS import HRViTRS
from models.SegViT_RS import SegViTRS

from tr_new2 import trainer_synapse


# 定义自定义参数解析
def parse_args():
    parser = argparse.ArgumentParser()

    # 基础配置
    parser.add_argument('--root_path', type=str,
                        default='/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved',
                        help='数据根目录')
    parser.add_argument('--dataset', type=str, default='遥感图像语义分割对比实验', help='实验名称')
    parser.add_argument('--list_dir', type=str,
                        default='/root/autodl-tmp/ST-Unet/datasets/Vaihingen/lists_txt_RGB_improved',
                        help='数据列表目录')
    parser.add_argument('--num_classes', type=int, default=6, help='输出类别数')
    parser.add_argument('--max_iterations', type=int, default=30000, help='最大迭代数')
    parser.add_argument('--max_epochs', type=int, default=150, help='最大训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--n_gpu', type=int, default=1, help='GPU数量')
    parser.add_argument('--deterministic', type=int, default=1, help='确定性训练')
    parser.add_argument('--base_lr', type=float, default=0.001, help='基础学习率')
    parser.add_argument('--img_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--n_skip', type=int, default=3, help='跳跃连接数')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT模型名称')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT补丁尺寸')
    parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None, help='注意力类型')

    # 新增：TransUNet 专属预训练权重路径（极其致命，必须有！）
    parser.add_argument('--pretrained_path', type=str,
                        default='/root/autodl-tmp/ST-Unet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
                        help='ViT预训练权重绝对路径')

    # 新增：正则化参数
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--drop_path_rate', type=float, default=0.25, help='Drop Path率')

    # 模型选择
    parser.add_argument('--model_name', type=str,
                        default='HRViTRS',
                        choices=['SegViTRS','HRViTRS'],
                        help='选择训练的模型')

    # 数据统计
    parser.add_argument('--data_stats_path', type=str,
                        default="/root/autodl-tmp/ST-Unet/datasets/Vaihingen/rgb_data_stats_improved.npz",
                        help='数据统计文件路径')

    return parser.parse_args()


def get_model(args):
    """根据模型名称获取模型"""
    if args.model_name == 'SegViTRS':
        cfg = {
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "patch_size": 4,
            "window_size": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.1,
            "use_geo_pos_encoding": True,
            "use_land_prior": True,
            "decoder_embed_dim": 64
        }
        return SegViTRS(cfg=cfg, img_size=args.img_size).cuda()

    elif args.model_name == 'HRViTRS':
        CFG = {
            "in_channels": 3,  # 输入通道数（RGB遥感图像）
            "num_classes": 6,  # 6分类输出
            "embed_dims": [64, 128],  # 高分辨率流的嵌入维度
            "num_heads": [4, 8],  # 注意力头数
            "window_size": 4,  # 遥感适配窗口大小（4×4）
            "depths": [2, 2],  # 各分辨率流的HRViT块数量
            "drop_rate": 0.1,  # Dropout率
            "use_spectral_attention": True,  # 遥感专属：光谱注意力
            "use_geo_pos_encoding": True  # 遥感专属：地理空间位置编码
        }
        return HRViTRS(CFG).cuda()
    else:
        raise ValueError(f"未支持的模型: {args.model_name}")


def get_trainer(args):
    """获取训练器"""
    if args.model_name == 'SegViTRS_Balanced':
        return trainer_synapse
    else:
        return trainer_synapse


def main():
    args = parse_args()

    # 确定性训练配置
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置
    args.img_size = 256
    args.batch_size = 8
    dataset_name = 'Vai_256'

    dataset_config = {
        'Vai_256': {
            'root_path': '/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved',
            'list_dir': '/root/autodl-tmp/ST-Unet/datasets/Vaihingen/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        },
        'Pots_256':{
            'root_path': '/root/autodl-tmp/ST-Unet/datasets/Potsdam/npz_data_RGB_improved',
            'list_dir': '/root/autodl-tmp/ST-Unet/datasets/Potsdam/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        }
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_channels = dataset_config[dataset_name]['in_channels']

    # 强制开启预训练开关
    args.is_pretrain = True

    # 加载数据统计
    if not os.path.exists(args.data_stats_path):
        raise FileNotFoundError(f"数据统计文件不存在：{args.data_stats_path}")

    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()

    print("=" * 50)
    print(f"✅ 加载数据统计完成：")
    print(f"   均值（R/G/B）：{[round(x, 4) for x in train_mean]}")
    print(f"   标准差（R/G/B）：{[round(x, 4) for x in train_std]}")
    print("=" * 50)

    # 动态生成保存路径，修复硬编码 Pots 的问题
    dataset_prefix = dataset_name.split('_')[0]
    snapshot_path = os.path.join(
        "/root/autodl-tmp/ST-Unet/ComparedModels_U",
        f"{args.model_name}_{dataset_prefix}_networksD",
        f"{args.model_name}_{dataset_name}_{args.img_size}",
        f"iter{args.max_iterations // 1000}k_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}"
    )

    os.makedirs(snapshot_path, exist_ok=True)

    print(f"📁 模型保存路径: {snapshot_path}")
    print("=" * 50)

    # 获取模型
    print(f"🔧 初始化模型: {args.model_name}")
    net = get_model(args)

    # 计算并显示模型参数
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"📊 模型参数量:")
    print(f"   总参数量: {total_params / 1e6:.2f} M")
    print(f"   可训练参数量: {trainable_params / 1e6:.2f} M")
    print("=" * 50)

    # 检查点恢复
    start_epoch = 0
    latest_ckpt_path = None

    if os.path.exists(snapshot_path):
        ckpt_files = [f for f in os.listdir(snapshot_path)
                      if f.startswith('RGBepoch_') and f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            latest_ckpt = ckpt_files[0]
            latest_ckpt_path = os.path.join(snapshot_path, latest_ckpt)
            print(f"找到{args.model_name}的最新Checkpoint: {latest_ckpt_path}")

            checkpoint = torch.load(latest_ckpt_path, map_location=torch.device('cuda'))
            net.load_state_dict(checkpoint['model'])
            current_epoch = int(latest_ckpt.split('_')[1].split('.')[0])
            start_epoch = current_epoch
            print(f"已恢复模型状态，将从epoch {start_epoch} 继续训练")
        else:
            print(f"未找到{args.model_name}的历史Checkpoint，将从头开始训练")
    else:
        print("快照路径不存在，将从头开始训练并创建路径")

    # 获取训练器
    trainer = get_trainer(args)

    # 打印训练参数
    print(f"🎯 训练参数:")
    print(f"   模型: {args.model_name}")
    print(f"   学习率: {args.base_lr}")
    print(f"   批大小: {args.batch_size}")
    print(f"   Epochs: {args.max_epochs}")
    print(f"   随机种子: {args.seed}")
    print("=" * 50)

    # 启动训练
    print("🚀 开始训练...")
    print("=" * 50)

    trainer(
        args,
        net,
        snapshot_path,
        start_epoch=start_epoch,
        train_mean=train_mean,
        train_std=train_std
    )


if __name__ == "__main__":
    main()