import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FCN(nn.Module):
    def __init__(self, num_classes=6, in_channels=3):  # 默认改为3通道（对齐训练代码）
        super(FCN, self).__init__()
        # ===================== 核心修改1：移除预训练权重，适配3通道 =====================
        # 加载ResNet50主干，禁用预训练（避免3通道预训练权重与自定义通道冲突）
        backbone = resnet50(weights=None)  # 改为None，避免加载ImageNet预训练权重
        # 动态替换第一层卷积，适配输入通道数
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # 初始化新的卷积层权重（提升训练收敛性）
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        # ==============================================================================

        # 提取各阶段特征（保持ResNet50的层级结构）
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 替换全连接为卷积（移除bias，配合BN更稳定）
        self.fcn = nn.Conv2d(2048, num_classes, kernel_size=1, bias=False)
        # 初始化分割头权重
        nn.init.kaiming_normal_(self.fcn.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 保存原始输入尺寸（动态适配任意输入尺寸，不再固定256×256）
        original_size = (x.shape[2], x.shape[3])

        # 主干网络提取特征
        x = self.layer1(x)  # (B, 256, H/4, W/4)
        x = self.layer2(x)  # (B, 512, H/8, W/8)
        x = self.layer3(x)  # (B, 1024, H/16, W/16)
        x = self.layer4(x)  # (B, 2048, H/32, W/32)

        # 1x1卷积降维到类别数
        x = self.fcn(x)  # (B, num_classes, H/32, W/32)

        # ===================== 核心修改2：动态上采样到原始尺寸 =====================
        # 不再固定256×256，适配任意输入尺寸（如128/256/512）
        x = F.interpolate(
            x, size=original_size,
            mode='bilinear', align_corners=False
        )
        return x


# 完整测试代码（验证维度、梯度、鲁棒性）
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型（与训练代码参数完全一致）
    model = FCN(
        num_classes=6,
        in_channels=3  # 训练代码的RGB 3通道
    ).to(device)

    # 测试1：标准256×256输入（训练代码常用尺寸）
    test_input = torch.randn(8, 3, 256, 256).to(device)  # batch_size=8
    output = model(test_input)
    print("\n=== 标准尺寸测试 ===")
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 预期：torch.Size([8, 6, 256, 256])
    assert output.shape == (8, 6, 256, 256), "标准尺寸输出不匹配！"

    # 测试2：非标准尺寸鲁棒性（如512×512）
    test_input_large = torch.randn(2, 3, 512, 512).to(device)
    output_large = model(test_input_large)
    print("\n=== 非标准尺寸测试 ===")
    print(f"输入尺寸: {test_input_large.shape}")
    print(f"输出尺寸: {output_large.shape}")  # 预期：torch.Size([2, 6, 512, 512])
    assert output_large.shape == (2, 6, 512, 512), "非标准尺寸输出不匹配！"

    # 测试3：梯度传播验证
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    target = torch.randint(0, 6, (8, 256, 256)).to(device)
    loss = loss_fn(output, target)
    loss.backward()

    # 检查关键层梯度
    grad_ok = all([
        model.fcn.weight.grad is not None,
        model.layer4[0].conv1.weight.grad is not None
    ])
    print("\n=== 梯度传播测试 ===")
    print(f"梯度传播: {'✅ 通过' if grad_ok else '❌ 失败'}")
    assert grad_ok, "梯度传播失败！"

    # 测试4：参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== 参数量统计 ===")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")

    print("\n✅ 所有测试通过！")