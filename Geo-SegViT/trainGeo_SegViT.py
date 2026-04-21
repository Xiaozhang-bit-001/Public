import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

# 设置GPU（单卡对比，确保公平性）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入自定义模型（你的基础模型）
from modelingnew import CONFIGS as CONFIGS_ViT_seg

# 导入对比模型（需确保models文件夹下有对应文件）
from models.FCN import FCN
from models.UNet import UNet
from models.SegViT_RS import SegViTRS
from models.Geo_SegViT import Geo_SegViT
from models.UperNet import UperNet
from models.DANet import DANet
from models.TransUnet import TransUNet
from models.SwinUnet import SwinUNet
from models.DeepLabVp import DeeplabV3Plus

# 导入训练函数（确保trainer_synapse签名支持 train_mean/train_std 参数）
from tr_new2 import trainer_synapse


parser = argparse.ArgumentParser()
# 基础配置（所有模型共用，确保对比公平）
parser.add_argument('--root_path', type=str,
                    default=r'G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Potsdam\npz_data_RGB_improved',
                    help='数据根目录')
parser.add_argument('--dataset', type=str, default='遥感图像语义分割对比实验', help='实验名称')
parser.add_argument('--list_dir', type=str,
                    default=r'G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Potsdam\lists_txt_RGB_improved',
                    help='数据列表目录')
parser.add_argument('--num_classes', type=int, default=6, help='输出类别数')
parser.add_argument('--max_iterations', type=int, default=30000, help='最大迭代数')
parser.add_argument('--max_epochs', type=int, default=150, help='最大训练轮数')
parser.add_argument('--batch_size', type=int, default=8, help='单卡batch size（命令行指定时生效）')
parser.add_argument('--n_gpu', type=int, default=1, help='GPU数量（固定单卡）')
parser.add_argument('--deterministic', type=int, default=1, help='是否启用确定性训练')
parser.add_argument('--base_lr', type=float, default=0.01, help='学习率（所有模型统一）')
parser.add_argument('--img_size', type=int, default=256, help='输入图像尺寸（所有模型统一）')
parser.add_argument('--seed', type=int, default=1234, help='随机种子（确保可复现）')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None, help='注意力类型（特定模型用）')

# 新增：模型选择参数（核心改造）
parser.add_argument('--model_name', type=str,
                    default='Geo_SegViT',  # 默认训练FCN
                    choices=[ 'FCN', 'UNet','SegViTRS','Geo_SegViT', 'UperNet', 'DANet', 'TransUNet', 'SwinUNet', 'DeeplabV3Plus'],
                    help='选择训练的模型')

# 数据统计文件路径（所有模型共用同一统计数据）
parser.add_argument('--data_stats_path', type=str,
                    default=r"G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Potsdam\rgb_data_stats_improved.npz",
                    help='数据均值/标准差文件路径')

args = parser.parse_args()


if __name__ == "__main__":
    # 确定性训练配置（所有模型保持一致）
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 固定随机种子（确保实验可复现）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置（统一路径和参数，所有模型共用）
    args.img_size = 256  # 强制统一图像尺寸
    args.batch_size = 8  # 强制统一batch size（覆盖命令行输入，确保对比公平）
    dataset_name = 'Pots_256'  # 换成自己的'
    dataset_config = {
        'Vai_256': {
            'root_path': r'G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Vaihingen\npz_data_RGB_improved',
            'list_dir': r'G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Vaihingen\lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3  # Vaihingen为RGB 3通道
        },
        'Pots_256': {
            'root_path': r'G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Potsdam\npz_data_RGB_improved',
            'list_dir': r'G:\erya308\zhangjunming\modeL\ST-Unet\datasets\Potsdam\lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3  # Potsdam为RGB 3通道
        },
    }
    # 更新数据集参数（覆盖命令行输入，确保与数据集匹配）
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_channels = dataset_config[dataset_name]['in_channels']
    args.is_pretrain = False  # 统一不使用预训练（如需开启，改为True并确保模型支持）

    # 加载数据统计（所有模型共用同一归一化参数）
    if not os.path.exists(args.data_stats_path):
        raise FileNotFoundError(f"数据统计文件不存在：{args.data_stats_path}")
    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()
    print(f"✅ 加载数据统计完成（所有模型共用）：")
    print(f"   均值（R/G/B）：{[round(x, 4) for x in train_mean]}")
    print(f"   标准差（R/G/B）：{[round(x, 4) for x in train_std]}")

    # -------------------------- 核心修复：快照路径构建（简洁、动态、无重复）--------------------------
    # 1. 动态模型文件夹（根据model_name自动生成，避免硬编码FCNnetworks）
    model_folder = f"{args.model_name}_Pot_networks"
    # 2. 核心标识（模型+数据集+图像尺寸）
    core_tag = f"{args.model_name}_{dataset_name}_{args.img_size}"
    # 3. ViT专用参数标签（其他模型为空）
    vit_tag = ""
    if args.model_name == 'ViT_seg':
        vit_tag = f"vit{args.vit_name}_skip{args.n_skip}_patch{args.vit_patches_size}"
    # 4. 通用超参数标签（迭代数、epoch、batch、lr、种子）
    iter_tag = f"iter{args.max_iterations//1000}k"  # 30000 → iter30k
    epoch_tag = f"epo150"
    batch_tag = f"bs{args.batch_size}"
    lr_tag = f"lr{args.base_lr}"
    seed_tag = f"s{args.seed}"
    param_tag = "_".join([iter_tag, epoch_tag, batch_tag, lr_tag, seed_tag])

    # 5. 拼接完整路径（用os.path.join自动处理分隔符，避免重复）
    snapshot_path = os.path.join(
        "G:\erya308\zhangjunming\modeL\ST-Unet\ComparedModel2_Best",  # 根目录
        model_folder,  # 模型专属文件夹（如FCN_networks）
        core_tag,      # 核心标识
        vit_tag,       # ViT专用参数（其他模型为空）
        param_tag      # 通用超参数
    )
    # 清理路径中的空字符串（避免ViT外的模型出现多余的"/"）
    snapshot_path = os.path.normpath(snapshot_path)
    # ---------------------------------------------------------------------------------------------

    print('-------------------------------------------')
    print(f"当前模型: {args.model_name}")
    print(f"快照保存路径: {snapshot_path}")
    print('-------------------------------------------')
    # 创建路径（支持多级目录自动创建）
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path, exist_ok=True)

    # 初始化模型（根据model_name选择，保持原逻辑）
    if  args.model_name == 'FCN':
        net = FCN(
            num_classes=args.num_classes,
            in_channels=args.in_channels
        ).cuda()

    elif args.model_name == 'SegViTRS':
        cfg = {
            "in_channels": args.in_channels,  # 3
            "num_classes": args.num_classes,  # 6
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

        net = SegViTRS(
            cfg=cfg,
            img_size=args.img_size
        ).cuda()

    elif args.model_name == 'Geo_SegViT':
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

        net = Geo_SegViT(
            cfg=cfg,
            img_size=args.img_size
        ).cuda()

    elif args.model_name == 'UNet':
        net = UNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            features=[64, 128, 256, 256]  # 统一通道设置
        ).cuda()

    elif args.model_name == 'DeeplabV3Plus':
        net = DeeplabV3Plus(
            num_classes=args.num_classes,
            in_channels=args.in_channels
        ).cuda()

    elif args.model_name == 'UperNet':
        net = UperNet(
            num_classes=args.num_classes,
            in_channels=args.in_channels
        ).cuda()

    elif args.model_name == 'DANet':
        net = DANet(
            num_classes=args.num_classes,
            in_channels=args.in_channels
        ).cuda()


    elif args.model_name == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]  # 加载ViT配置（hidden_size=768）
        config_vit.n_classes = args.num_classes
        net = TransUNet(
            config=config_vit,  # 必须传递config，确保hidden_size统一
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            img_size=args.img_size
        ).cuda()

    elif args.model_name == 'SwinUNet':
        net = SwinUNet(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            img_size=args.img_size,
            embed_dim=96,
            depths=tuple([2, 2, 6, 2]),  # 列表转元组，避免类型错误
            num_heads=tuple([3, 6, 12, 24])
        ).cuda()

    else:
        raise ValueError(f"未支持的模型: {args.model_name}")

    # 断点续训逻辑（保持原逻辑，适配新路径）
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
            start_epoch = current_epoch  # 修复：原逻辑多+1，导致跳过一个epoch（如保存的是epoch112，应从112继续）
            print(f"已恢复模型状态，将从epoch {start_epoch} 继续训练")
        else:
            print(f"未找到{args.model_name}的历史Checkpoint，将从头开始训练")
    else:
        print("快照路径不存在，将从头开始训练并创建路径")

    # 启动训练（保持原参数传递方式，train_mean/train_std已适配RandomGenerator）
    trainer_synapse(
        args,
        net,
        snapshot_path,
        start_epoch=start_epoch,
        train_mean=train_mean,
        train_std=train_std
    )