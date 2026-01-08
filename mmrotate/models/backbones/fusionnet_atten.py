# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from kymatio.torch import Scattering2D
from mmrotate.registry import MODELS


@MODELS.register_module()
class FusionNet_Atten(BaseModule):
    def __init__(self, backbone, backbone2, input_size=(800, 800), init_cfg=None, atten = 'spatial', Wave_J = 1):
        super().__init__(init_cfg=init_cfg)
        if atten == 'spatial_channel':
            ChannelAttentionFusion = SpatialChannelAttentionFusion
        elif atten == 'Cross':
            ChannelAttentionFusion = SimplifiedCrossAttention
        elif atten == 'Gate':
            ChannelAttentionFusion = SARSpecializedFusion
        self.input_size = input_size
        self.wavelet_trans = Scattering2D(J=Wave_J, shape=self.input_size)
        backbone['in_channels'] = 3
        if Wave_J == 1:
            backbone2['in_channels'] = 9
        elif Wave_J == 2:
            backbone2['in_channels'] = 81
        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone2)

        # ResNet50 的实际通道数
        self.channels_list = [256, 512, 1024, 2048]
        self.attention_fusions = nn.ModuleList()

        for channels in self.channels_list:
            self.attention_fusions.append(
                ChannelAttentionFusion(channels, reduction=16)
            )

    def forward(self, x):
        x_ = x.mean(1, keepdim=True)
        with torch.no_grad():
            x_ = nn.functional.interpolate(
                self.wavelet_trans(x_).squeeze(1), self.input_size, mode='bilinear'
            )

        features1 = self.backbone(x)
        features2 = self.backbone2(x_)

        fused_features = []
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            fused = self.attention_fusions[i](f1, f2)
            fused_features.append(fused)

        return tuple(fused_features)


class SpatialChannelAttentionFusion(nn.Module):
    """结合空间和通道注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # 1. 通道注意力（改进版）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            nn.Sigmoid()
        )

        # 2. 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),  # 7x7卷积捕获大感受野
            nn.Sigmoid()
        )

        # 3. 残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        batch_size = x1.shape[0]

        # 通道注意力权重
        channel_weights = self.channel_attention(concat)  # (B, 2C, 1, 1)
        channel_weights = channel_weights.view(batch_size, 2, self.channels, 1, 1)

        # 空间注意力权重
        avg_out = torch.mean(concat, dim=1, keepdim=True)
        max_out, _ = torch.max(concat, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)  # (B, 1, H, W)

        # 融合：通道权重 × 空间权重
        weight1 = channel_weights[:, 0] * spatial_weights
        weight2 = channel_weights[:, 1] * spatial_weights

        # 加权融合 + 残差
        fused = x1 * weight1 + x2 * weight2
        residual = (x1 + x2) * self.residual_scale

        return fused + residual


class SimplifiedCrossAttention(nn.Module):
    """先交互再生成权重，计算量适中"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # 1. 特征交互层
        self.interaction = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 2. 权重生成（简化版，生成全局权重）
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
            nn.Sigmoid()
        )

        # 3. 可选的逐位置权重（更细粒度但更重）
        self.use_spatial = False
        if self.use_spatial:
            self.spatial_weight = nn.Sequential(
                nn.Conv2d(channels * 2, 2, 3, padding=1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)

        # 特征交互
        interacted = self.interaction(concat)

        # 生成权重
        if self.use_spatial:
            # 逐位置权重（计算量大）
            spatial_weights = self.spatial_weight(concat)  # (B, 2, H, W)
            weight1, weight2 = spatial_weights[:, 0:1], spatial_weights[:, 1:2]
            fused = x1 * weight1 + x2 * weight2
        else:
            # 全局权重（你的原始版本改进）
            global_weights = self.weight_gen(interacted)  # (B, 2C, 1, 1)
            global_weights = global_weights.view(-1, 2, self.channels, 1, 1)
            weight1, weight2 = global_weights[:, 0], global_weights[:, 1]
            fused = x1 * weight1 + x2 * weight2

        return fused


class SARSpecializedFusion(nn.Module):
    """考虑SAR图像特性：强散射点、边缘清晰"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        # 1. 边缘敏感模块（给小波特征）
        self.edge_sensitive = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 深度可分离
            nn.Conv2d(channels, channels, 1),  # 点卷积
            nn.Sigmoid()
        )

        # 2. 区域敏感模块（给原始特征）
        self.region_sensitive = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # 3. 门控机制决定主导分支
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # x1: 原始特征，x2: 小波特征

        # 边缘增强（小波更擅长）
        edge_enhanced = x2 * self.edge_sensitive(x2)

        # 区域增强（原始更擅长）
        region_enhanced = x1 * self.region_sensitive(x1)

        # 自适应门控
        concat = torch.cat([x1, x2], dim=1)
        gate_weights = self.gate(concat)  # (B, 2, 1, 1)
        gate1, gate2 = gate_weights[:, 0:1], gate_weights[:, 1:2]

        # 融合
        fused = region_enhanced * gate1 + edge_enhanced * gate2

        return fused