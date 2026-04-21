import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class PositionAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

class DANet(nn.Module):
    def __init__(self, num_classes=6, in_channels=4):
        super(DANet, self).__init__()
        # Backbone
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # 双注意力模块
        self.pos_att = PositionAttention(2048)
        self.chan_att = ChannelAttention(2048)
        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(2048 * 2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        pos_out = self.pos_att(x)
        chan_out = self.chan_att(x)
        x = torch.cat([pos_out, chan_out], dim=1)
        x = self.seg_head(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x