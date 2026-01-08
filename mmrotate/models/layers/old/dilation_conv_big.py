import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class MultiScaleDilationConv(nn.Module):
    """多尺度空洞卷积模块"""

    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3, 6]):
        super().__init__()

        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                ConvModule(
                    in_channels,
                    out_channels // 4,  # 每个分支输出1/4通道
                    3,
                    stride=1,
                    padding=rate,
                    dilation=rate,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                )
            )

        # 融合卷积
        self.fusion_conv = ConvModule(
            out_channels,  # 4个分支concat后就是out_channels
            out_channels,
            1,  # 1×1卷积融合特征
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # 在通道维度拼接
        out = torch.cat(branch_outputs, dim=1)
        out = self.fusion_conv(out)
        return out