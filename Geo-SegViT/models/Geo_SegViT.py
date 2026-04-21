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

# --------------------------- 改进：编码器层特征融合注意力模块 ---------------------------
class CrossLayerFeatureFusion(nn.Module):
    """编码器每一层输出 → 通道+空间双注意力 → 融合高质量特征"""
    def __init__(self, channels: int):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        x_channel = x * self.channel_att(x)
        # 空间注意力
        avg = torch.mean(x_channel, dim=1, keepdim=True)
        max, _ = torch.max(x_channel, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max], dim=1)
        x_spatial = x_channel * self.spatial_att(x_cat)
        return x_spatial

# --------------------------- ViT 基础模块 ---------------------------
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

# --------------------------- 改进版编码器 ---------------------------
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

        # 改进：每层增加特征融合模块
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
            # 改进：编码器每层输出 → 特征融合注意力
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

# --------------------------- 改进：Swin Transformer 式上采样 ---------------------------
class SwinUpsample(nn.Module):
    """Swin 反窗口像素重组上采样（替代传统卷积上采样）"""
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

# --------------------------- 改进：解码器 Transformer 块（无普通卷积） ---------------------------
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

# --------------------------- 改进版解码器 ---------------------------
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
            # Swin 上采样
            self.upsamples.append(SwinUpsample(current_dim))
            # 跳跃连接融合
            fuse_dim = current_dim + stage_channels[i-1]
            self.fusion_blocks.append(nn.Linear(fuse_dim, dec_dim))
            # Transformer 解码块
            self.decoder_blocks.append(DecoderTransformerBlock(dec_dim))
            current_dim = dec_dim

    def forward(self, features: List[torch.Tensor], hs: List[int], ws: List[int]):
        x = features[-1]
        for i in range(len(self.upsamples)):
            # 上采样
            x = self.upsamples[i](x)
            # 跳跃特征
            skip = features[-(i+2)]
            # 尺寸对齐
            x = F.interpolate(x, size=skip.shape[2:])
            # 融合
            x = rearrange(x, 'B C H W -> B (H W) C')
            skip = rearrange(skip, 'B C H W -> B (H W) C')
            x = torch.cat([x, skip], dim=-1)
            x = self.fusion_blocks[i](x)
            # Transformer 解码
            h, w = hs[-(i+2)], ws[-(i+2)]
            x = self.decoder_blocks[i](x, h, w)
            x = rearrange(x, 'B (H W) C -> B C H W', H=h, W=w)
        return x

# --------------------------- 改进：增强型分割头 ---------------------------
class EnhancedSegHead(nn.Module):
    def __init__(self, embed_dim, num_classes, patch_size):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear')
        # 深度可分离卷积
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.pwconv = nn.Conv2d(embed_dim, embed_dim//2, 1)
        self.norm = nn.BatchNorm2d(embed_dim//2)
        self.act = nn.GELU()
        # 输出层
        self.out = nn.Conv2d(embed_dim//2, num_classes, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.act(self.norm(x))
        return self.out(x)

# --------------------------- 完整改进模型 ---------------------------
class Geo_SegViT(nn.Module):
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

# --------------------------- 测试 ---------------------------
if __name__ == "__main__":
    model = Geo_SegViT()
    model.eval()
    test_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(test_input)
    print(f"输入: {test_input.shape}")
    print(f"输出: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(" 初始化成功！")