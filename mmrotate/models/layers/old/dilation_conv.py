import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class MultiScaleDilationConv(nn.Module):
    """轻量多尺度 - 小目标友好"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 只使用两个尺度，避免过大感受野
        self.branch1 = ConvModule(in_channels, out_channels // 2, 3, padding=1, dilation=1)
        self.branch2 = ConvModule(in_channels, out_channels // 2, 3, padding=2, dilation=2)

        # 可选：添加注意力机制增强小目标特征
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(out_channels, out_channels // 4, 1),
            ConvModule(out_channels // 4, out_channels, 1, act_cfg=dict(type='Sigmoid'))
        )

    def forward(self, x):
        b1 = self.branch1(x)  # 细节特征
        b2 = self.branch2(x)  # 上下文特征
        out = torch.cat([b1, b2], dim=1)

        # 空间注意力增强重要区域
        attention = self.se(out)
        return out * attention
