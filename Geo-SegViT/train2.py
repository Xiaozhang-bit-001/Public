import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import json
from datetime import datetime

# 设置GPU（单卡对比，确保公平性）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ======================== 导入新模型（核心新增） ========================
# 导入自定义的Res16_DualDecoder_SegModel
from model2.Mnew import Res16_DualDecoder_SegModel, SwinConfig

# 导入原有模型
from modelingnew import CONFIGS as CONFIGS_ViT_seg
from model2.UNet import UNet
from model2.UperNet import UperNet
from model2.TransUnet import TransUNet
from model2.SwinUnet import SwinUNet
from model2.DeepLabVp import DeeplabV3Plus

# 导入训练函数
from tr_new2 import trainer_synapse


# ======================== 日志配置 ========================
def setup_logger(args, dataset_name):
    """配置日志记录，同时输出到控制台和文件"""
    log_dir = os.path.join(args.final_result_dir, args.model_name, dataset_name)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(log_file, encoding='utf-8')  # 文件输出
        ]
    )
    return logging.getLogger(__name__)


# ======================== 参数解析 ========================
parser = argparse.ArgumentParser()
# 基础配置（所有模型共用）
parser.add_argument('--root_path', type=str,
                    default=r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/npz_data_RGB_improved',
                    help='数据根目录')
parser.add_argument('--dataset', type=str, default='遥感图像语义分割对比实验', help='实验名称')
parser.add_argument('--list_dir', type=str,
                    default=r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/lists_txt_RGB_improved',
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
parser.add_argument('--n_skip', type=int, default=3, help='跳跃连接数（仅用于ViT_seg）')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT模型名称（仅用于ViT_seg）')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT补丁尺寸（仅用于ViT_seg）')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None, help='注意力类型（特定模型用）')

# 模型选择参数（新增Res16_DualDecoder）
parser.add_argument('--model_name', type=str,
                    default='Res16_DualDecoder',  # 默认训练新模型
                    choices=['UNet', 'UperNet', 'TransUNet',
                             'SwinUNet', 'DeeplabV3Plus', 'Res16_DualDecoder'],  # 新增选项
                    help='选择训练的模型')

# 数据统计文件路径
parser.add_argument('--data_stats_path', type=str,
                    default=r"/root/autodl-tmp/ST-Unet/datasets/Potsdam/rgb_data_stats_improved.npz",
                    help='数据均值/标准差文件路径')

# 最终结果保存配置
parser.add_argument('--final_result_dir', type=str,
                    default=r"/root/autodl-tmp/ST-Unet/ComparedModel2/FinalResults_U",
                    help='最终结果保存根目录')
parser.add_argument('--save_best_only', action='store_true', default=False,
                    help='是否只保存最优模型（否则保存最终epoch模型）')

# 新模型专用参数（可选，如需自定义Swin配置）
parser.add_argument('--swin_embed_dim', type=int, default=64, help='Swin解码器嵌入维度（仅Res16_DualDecoder用）')
parser.add_argument('--swin_window_size', type=int, default=4, help='Swin窗口大小（仅Res16_DualDecoder用）')

args = parser.parse_args()


# ======================== 结果保存函数 ========================
def save_final_results(args, net, snapshot_path, dataset_name, train_mean, train_std, final_metrics=None):
    """保存最终训练结果：模型权重、配置文件、性能指标"""
    # 1. 创建最终结果目录
    final_dir = os.path.join(args.final_result_dir, args.model_name, dataset_name)
    os.makedirs(final_dir, exist_ok=True)

    # 2. 生成唯一标识
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    core_info = f"{args.model_name}_{dataset_name}_{args.img_size}_bs{args.batch_size}_ep{args.max_epochs}"

    # 3. 保存模型权重
    try:
        if args.save_best_only:
            # 查找最优模型
            best_ckpt_files = [f for f in os.listdir(snapshot_path) if f.startswith('best_')]
            if best_ckpt_files:
                best_ckpt = sorted(best_ckpt_files)[-1]
                best_ckpt_path = os.path.join(snapshot_path, best_ckpt)
                final_ckpt_path = os.path.join(final_dir, f"{core_info}_best_{timestamp}.pth")
                shutil.copy2(best_ckpt_path, final_ckpt_path)
                logging.info(f"✅ 已保存最优模型到：{final_ckpt_path}")
            else:
                logging.warning("未找到最优模型文件，保存最终epoch模型")
                final_model_path = os.path.join(final_dir, f"{core_info}_final_{timestamp}.pth")
                torch.save({
                    'model': net.state_dict(),
                    'epoch': args.max_epochs,
                    'args': args,
                    'final_metrics': final_metrics,
                    'train_finish_time': timestamp
                }, final_model_path)
                logging.info(f"✅ 已保存最终模型权重到：{final_model_path}")
        else:
            # 保存最终epoch模型
            final_model_path = os.path.join(final_dir, f"{core_info}_final_{timestamp}.pth")
            torch.save({
                'model': net.state_dict(),
                'epoch': args.max_epochs,
                'args': args,
                'final_metrics': final_metrics,
                'train_finish_time': timestamp
            }, final_model_path)
            logging.info(f"✅ 已保存最终模型权重到：{final_model_path}")
    except Exception as e:
        logging.error(f"保存模型权重失败：{str(e)}")
        raise

    # 4. 保存训练配置
    try:
        config_path = os.path.join(final_dir, f"{core_info}_config_{timestamp}.json")
        serializable_args = {
            k: v for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, tuple))
        }
        serializable_args.update({
            'train_finish_time': timestamp,
            'dataset_stats': {'mean': train_mean, 'std': train_std},
            'dataset_name': dataset_name
        })
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_args, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ 已保存训练配置到：{config_path}")
    except Exception as e:
        logging.error(f"保存配置文件失败：{str(e)}")
        raise

    # 5. 保存性能指标
    if final_metrics is not None:
        try:
            metrics_path = os.path.join(final_dir, f"{core_info}_metrics_{timestamp}.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=4)
            logging.info(f"✅ 已保存性能指标到：{metrics_path}")
        except Exception as e:
            logging.error(f"保存性能指标失败：{str(e)}")
            raise

    return final_dir


# ======================== 主训练逻辑 ========================
if __name__ == "__main__":
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
    args.img_size = 256  # 强制统一图像尺寸
    args.batch_size = 8  # 强制统一batch size
    dataset_name = 'Pots_256'  # 可切换为'Pots_256'
    dataset_config = {
        'Vai_256': {
            'root_path': r'/root/autodl-tmp/ST-Unet/datasets/Vaihingen/npz_data_RGB_improved',
            'list_dir': r'/root/autodl-tmp/ST-Unet/datasets/Vaihingen/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        },
        'Pots_256': {
            'root_path': r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/npz_data_RGB_improved',
            'list_dir': r'/root/autodl-tmp/ST-Unet/datasets/Potsdam/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3
        },
    }
    # 更新数据集参数
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_channels = dataset_config[dataset_name]['in_channels']
    args.is_pretrain = False

    # 初始化日志
    logger = setup_logger(args, dataset_name)
    logger.info(f"开始训练 {args.model_name} 模型，数据集：{dataset_name}")

    # 加载数据统计
    if not os.path.exists(args.data_stats_path):
        raise FileNotFoundError(f"数据统计文件不存在：{args.data_stats_path}")
    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()
    logger.info(f"✅ 加载数据统计完成：")
    logger.info(f"   均值（R/G/B）：{[round(x, 4) for x in train_mean]}")
    logger.info(f"   标准差（R/G/B）：{[round(x, 4) for x in train_std]}")

    # 构建快照保存路径
    model_folder = f"{args.model_name}_networks"
    core_tag = f"{args.model_name}_{dataset_name}_{args.img_size}"
    vit_tag = ""
    if args.model_name == 'ViT_seg':
        vit_tag = f"vit{args.vit_name}_skip{args.n_skip}_patch{args.vit_patches_size}"
    # 新模型专用标签（可选）
    new_model_tag = ""
    if args.model_name == 'Res16_DualDecoder':
        new_model_tag = f"swin_embed{args.swin_embed_dim}_win{args.swin_window_size}"

    # 通用超参数标签
    iter_tag = f"iter{args.max_iterations // 1000}k"
    epoch_tag = f"epo{args.max_epochs}"
    batch_tag = f"bs{args.batch_size}"
    lr_tag = f"lr{args.base_lr}"
    seed_tag = f"s{args.seed}"
    param_tag = "_".join([iter_tag, epoch_tag, batch_tag, lr_tag, seed_tag])

    # 拼接路径（过滤空字符串）
    path_parts = [
        "/root/autodl-tmp/ST-Unet/ComResult_U",
        model_folder,
        core_tag,
        vit_tag,
        new_model_tag,
        param_tag
    ]
    path_parts = [p for p in path_parts if p.strip()]
    snapshot_path = os.path.normpath(os.path.join(*path_parts))

    logger.info('-------------------------------------------')
    logger.info(f"当前模型: {args.model_name}")
    logger.info(f"快照保存路径: {snapshot_path}")
    logger.info('-------------------------------------------')
    os.makedirs(snapshot_path, exist_ok=True)

    # ======================== 模型初始化（核心新增） ========================
    if  args.model_name == 'UNet':
        net = UNet(in_channels=args.in_channels, num_classes=args.num_classes, features=[64, 128, 256, 256]).cuda()

    elif args.model_name == 'DeeplabV3Plus':
        net = DeeplabV3Plus(num_classes=args.num_classes, in_channels=args.in_channels).cuda()

    elif args.model_name == 'UperNet':
        net = UperNet(num_classes=args.num_classes, in_channels=args.in_channels).cuda()

    elif args.model_name == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        net = TransUNet(config=config_vit, num_classes=args.num_classes,
                        in_channels=args.in_channels, img_size=args.img_size).cuda()

    elif args.model_name == 'SwinUNet':
        net = SwinUNet(num_classes=args.num_classes, in_channels=args.in_channels,
                       img_size=args.img_size, embed_dim=96,
                       depths=tuple([2, 2, 6, 2]), num_heads=tuple([3, 6, 12, 24])).cuda()

    # -------------------- 新模型初始化（核心新增） --------------------
    elif args.model_name == 'Res16_DualDecoder':
        # 初始化Swin配置（可通过命令行参数自定义）
        swin_config = SwinConfig()
        swin_config.embed_dim = args.swin_embed_dim
        swin_config.window_size = args.swin_window_size

        # 初始化新模型
        net = Res16_DualDecoder_SegModel(
            num_classes=args.num_classes,
            swin_config=swin_config
        ).cuda()
        logger.info(f"✅ 新模型Res16_DualDecoder初始化完成，Swin配置：")
        logger.info(f"   embed_dim: {swin_config.embed_dim}, window_size: {swin_config.window_size}")

    else:
        raise ValueError(f"未支持的模型: {args.model_name}")

    # ======================== 断点续训 ========================
    start_epoch = 0
    if os.path.exists(snapshot_path):
        ckpt_files = [f for f in os.listdir(snapshot_path)
                      if f.startswith('RGBepoch_') and f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            latest_ckpt = ckpt_files[0]
            latest_ckpt_path = os.path.join(snapshot_path, latest_ckpt)
            logger.info(f"找到{args.model_name}的最新Checkpoint: {latest_ckpt_path}")

            checkpoint = torch.load(latest_ckpt_path, map_location=torch.device('cuda'))
            net.load_state_dict(checkpoint['model'])
            current_epoch = int(latest_ckpt.split('_')[1].split('.')[0])
            start_epoch = current_epoch
            logger.info(f"已恢复模型状态，将从epoch {start_epoch} 继续训练")
        else:
            logger.info(f"未找到{args.model_name}的历史Checkpoint，从头开始训练")
    else:
        logger.info("快照路径不存在，创建路径并从头训练")

    # ======================== 启动训练 ========================
    logger.info(f"\n🚀 启动{args.model_name}模型训练！")
    logger.info(f"   数据集：{dataset_name} | 图像尺寸：{args.img_size}×{args.img_size}")
    logger.info(f"   Batch Size：{args.batch_size} | 学习率：{args.base_lr}")
    logger.info(f"   最大Epoch：{args.max_epochs} | 最大迭代数：{args.max_iterations}")

    # 执行训练
    final_metrics = trainer_synapse(
        args,
        net,
        snapshot_path,
        start_epoch=start_epoch,
        train_mean=train_mean,
        train_std=train_std
    )

    # ======================== 保存最终结果 ========================
    logger.info("\n" + "=" * 50)
    logger.info("开始保存最终训练结果...")
    logger.info("=" * 50)
    save_final_results(args, net, snapshot_path, dataset_name, train_mean, train_std, final_metrics)
    logger.info("\n🎉 所有最终结果保存完成！")