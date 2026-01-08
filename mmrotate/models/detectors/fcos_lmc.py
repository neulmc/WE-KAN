import math
from typing import Tuple, Union

import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from torch import Tensor
from torch.nn.functional import grid_sample
from mmdet.models.detectors.fcos import FCOS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
import torch.nn as nn
import torch.nn.functional as F

@MODELS.register_module()
class FCOS_typical(FCOS):
    def __init__(self,
                 backbone: ConfigType,
                 backbone2: ConfigType,
                 neck: ConfigType,
                 neck2: ConfigType,
                 bbox_head: ConfigType,
                 crop_size: Tuple[int, int] = (768, 768),
                 padding: str = 'reflection',
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 high_ratio: float = None,) -> None:

        self.backbone2_cfg = backbone2
        self.neck2_cfg = neck2
        self.high_ratio = high_ratio

        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # 构建第二个backbone和neck
        self.backbone2 = MODELS.build(backbone2)
        self.neck2 = MODELS.build(neck2)
        self.spatial_conv = nn.Conv2d(256, 1, 3, padding=1)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """重写特征提取方法，双backbone + 双neck"""

        # 生成高分辨率输入
        x_highres = torch.nn.functional.interpolate(
            batch_inputs, scale_factor=1/self.high_ratio, mode='bilinear', align_corners=False)

        # 提取特征
        features1 = self.backbone(batch_inputs)  # 原始分辨率特征
        features2 = self.backbone2(x_highres)  # 高分辨率特征

        # 分别通过各自的neck
        if self.with_neck:
            neck_features1 = self.neck(features1)  # 第一个neck输出
        else:
            neck_features1 = features1

        if hasattr(self, 'neck2'):
            neck_features2 = self.neck2(features2)  # 第二个neck输出
        else:
            neck_features2 = features2

        # 双neck特征融合：先相加再concat
        fused_features_feat1 = []
        fused_features_feat2 = []
        for i, (feat1, feat2) in enumerate(zip(neck_features1, neck_features2)):
            feat1_large = F.interpolate(feat1, size=feat2.shape[-2:], mode='bilinear')
            feat2_small = F.interpolate(feat2, size=feat1.shape[-2:], mode='bilinear')

            # 空间注意力融合
            # mixed_feat1: feat1为主，根据内容动态融合feat2_small
            spatial_weights1 = self._spatial_attention(feat1, feat2_small)
            mixed_feat1 = feat1 + spatial_weights1 * feat2_small

            # mixed_feat2: feat2为主，根据内容动态融合feat1_large
            spatial_weights2 = self._spatial_attention(feat2, feat1_large)
            mixed_feat2 = feat2 + spatial_weights2 * feat1_large

            fused_features_feat1.append(mixed_feat1)
            fused_features_feat2.append(mixed_feat2)

        return tuple(fused_features_feat1), tuple(fused_features_feat2)

    def _spatial_attention(self, main_feat, aux_feat):
        """空间注意力：根据每个位置的内容决定融合强度"""
        # 计算两个特征的差异
        diff = torch.abs(main_feat - aux_feat)
        # 通过卷积学习注意力权重
        attention = torch.sigmoid(self.spatial_conv(diff))
        return attention

    def forward(self, inputs: torch.Tensor, data_samples: SampleList = None,
                mode: str = 'tensor', **kwargs):
        """重写forward方法，适配MMDetection接口"""
        # 适配不同的输入格式
        if isinstance(inputs, dict):
            # 如果输入是字典，提取tensor和data_samples
            batch_inputs = inputs.get('inputs', inputs)
            data_samples = inputs.get('data_samples', data_samples)
        else:
            batch_inputs = inputs

        if mode == 'loss':
            return self.loss(batch_inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(batch_inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self.extract_feat(batch_inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList, **kwargs):
        """计算损失"""
        # 提取特征
        x1, x2 = self.extract_feat(batch_inputs)
        # 调用bbox_head计算损失
        losses = self.bbox_head.loss(x1 + x2, batch_data_samples, **kwargs)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x1, x2 = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x1 + x2, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples