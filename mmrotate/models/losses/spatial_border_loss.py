# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import QuadriBoxes

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS


@MODELS.register_module()
class RotatedL1Loss(nn.Module):
    """改进的旋转框L1损失，解决角度跃变问题。

    Args:
        loss_weight (float): 损失权重，默认1.0
        reduction (str): 损失归约方式，可选 'mean', 'sum', 'none'
        beta (float): Smooth L1的阈值，如果为1.0则是普通L1损失
        angle_version (str): 角度格式 'le90'或'oc'，默认'le90'
        use_relative_wh (bool): 宽高是否用相对误差，默认True
        normalize_angle (bool): 角度是否归一化，默认True
        balance_weights (list): 三部分损失权重 [xy, wh, angle]，默认[1.0, 1.0, 1.0]
        angle_method (str): 角度处理方法 'remainder'或'atan2'，默认'remainder' 应该只能用remainder
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 beta=1.0,
                 angle_version='le90',
                 use_relative_wh=True,
                 normalize_angle=True,
                 balance_weights=None,
                 angle_method='remainder'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.beta = beta
        self.angle_version = angle_version
        self.use_relative_wh = use_relative_wh
        self.normalize_angle = normalize_angle
        self.angle_method = angle_method

        # 检查参数
        assert reduction in ['mean', 'sum', 'none']
        assert beta > 0
        assert angle_version in ['le90', 'oc']
        assert angle_method in ['remainder', 'atan2']

        # 平衡权重 [xy, wh, angle]
        if balance_weights is None:
            self.balance_weights = [1.0, 1.0, 1.0]
        else:
            assert len(balance_weights) == 3
            self.balance_weights = balance_weights

        # 设置周期
        if angle_version == 'le90':
            self.period = 180.0  # [-90, 90] 实际周期180
            self.half_period = 90.0
        else:  # 'oc'
            self.period = 360.0  # [-180, 180] 周期360
            self.half_period = 180.0

    def _periodic_angle_diff(self, pred, target):
        """无跃变的角度差计算"""
        diff = pred - target

        if self.angle_method == 'remainder':
            # 方法1: remainder（快速稳定，无跃变）
            # 映射到 [-period/2, period/2)
            return torch.remainder(diff + self.half_period, self.period) - self.half_period

        else:  # 'atan2'
            # 方法2: atan2（最平滑，计算稍慢）
            # 转弧度
            diff_rad = torch.deg2rad(diff)
            # 计算最小角度差
            diff_rad = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))
            # 转回角度
            return torch.rad2deg(diff_rad)

    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """前向传播"""
        # 检查输入格式
        assert pred.shape[-1] == 5 and target.shape[-1] == 5, \
            f"输入应为5参数格式，得到 pred:{pred.shape[-1]}, target:{target.shape[-1]}"

        # 分离参数
        pred_xy, pred_wh, pred_angle = pred[:, :2], pred[:, 2:4], pred[:, 4]
        target_xy, target_wh, target_angle = target[:, :2], target[:, 2:4], target[:, 4]

        # 1. 中心点损失（Smooth L1）
        diff_xy = pred_xy - target_xy
        if self.beta == 1.0:
            loss_xy = torch.abs(diff_xy)
        else:
            loss_xy = torch.where(
                torch.abs(diff_xy) < self.beta,
                0.5 * diff_xy.pow(2) / self.beta,
                torch.abs(diff_xy) - 0.5 * self.beta
            )
        loss_xy = loss_xy.sum(dim=1) * self.balance_weights[0]

        # 2. 宽高损失
        diff_wh = pred_wh - target_wh
        eps = 1e-7  # 防止除零

        if self.use_relative_wh:
            # 相对误差：|pred-target|/(|target|+eps)
            # 加上eps防止除零，取绝对值防止负值
            denominator = target_wh.abs() + eps
            loss_wh = torch.abs(diff_wh) / denominator
        else:
            # 绝对误差
            if self.beta == 1.0:
                loss_wh = torch.abs(diff_wh)
            else:
                loss_wh = torch.where(
                    torch.abs(diff_wh) < self.beta,
                    0.5 * diff_wh.pow(2) / self.beta,
                    torch.abs(diff_wh) - 0.5 * self.beta
                )
        loss_wh = loss_wh.sum(dim=1) * self.balance_weights[1]

        # 3. 角度损失（无跃变）
        angle_diff = self._periodic_angle_diff(pred_angle, target_angle)

        if self.normalize_angle:
            # 归一化到[0, 1]范围
            loss_angle = torch.abs(angle_diff) / self.half_period
        else:
            loss_angle = torch.abs(angle_diff)

        loss_angle = loss_angle * self.balance_weights[2]

        # 总损失
        total_loss = loss_xy + loss_wh + loss_angle

        # 应用样本权重
        if weight is not None:
            assert weight.dim() == 1, f"权重应为1维，得到 {weight.dim()}维"
            assert weight.shape[0] == pred.shape[0], \
                f"权重数量 {weight.shape[0]} 与预测数量 {pred.shape[0]} 不匹配"
            total_loss = total_loss * weight

        # 归约
        if self.reduction == 'mean':
            if avg_factor is not None and avg_factor > 0:
                total_loss = total_loss.sum() / avg_factor
            else:
                total_loss = total_loss.mean()
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
        # reduction='none' 则保持不变

        return total_loss * self.loss_weight

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'loss_weight={self.loss_weight}, '
                f'reduction={self.reduction}, '
                f'beta={self.beta}, '
                f'angle_version={self.angle_version}, '
                f'use_relative_wh={self.use_relative_wh}, '
                f'normalize_angle={self.normalize_angle}, '
                f'balance_weights={self.balance_weights}, '
                f'angle_method={self.angle_method})')

@MODELS.register_module()
class KANIdenty(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KANIdenty, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, loss, *args, **kwargs):
        return loss * self.loss_weight

@MODELS.register_module()
class SpatialBorderLoss(nn.Module):
    """Spatial Border loss for learning points in Oriented RepPoints.

    Args:
        pts (torch.Tensor): point sets with shape (N, 9*2).
            Default points number in each point set is 9.
        gt_bboxes (torch.Tensor): gt_bboxes with polygon form with shape(N, 8)

    Returns:
        torch.Tensor: spatial border loss.
    """

    def __init__(self, loss_weight=1.0):
        super(SpatialBorderLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pts, gt_bboxes, weight, *args, **kwargs):
        loss = self.loss_weight * weighted_spatial_border_loss(
            pts, gt_bboxes, weight, *args, **kwargs)
        return loss


def spatial_border_loss(pts, gt_bboxes):
    """The loss is used to penalize the learning points out of the assigned
    ground truth boxes (polygon by default).

    Args:
        pts (torch.Tensor): point sets with shape (N, 9*2).
        gt_bboxes (torch.Tensor): gt_bboxes with polygon form with shape(N, 8)

    Returns:
        loss (torch.Tensor)
    """
    num_gts, num_pointsets = gt_bboxes.size(0), pts.size(0)
    num_point = int(pts.size(1) / 2.0)
    loss = pts.new_zeros([0])

    if num_gts > 0:
        inside_flag_list = []
        for i in range(num_point):
            pt = pts[:, (2 * i):(2 * i + 2)].reshape(num_pointsets,
                                                     2).contiguous()
            gt_qboxes = QuadriBoxes(gt_bboxes)
            inside_pt_flag = gt_qboxes.find_inside_points(pt, is_aligned=True)
            inside_flag_list.append(inside_pt_flag)

        inside_flag = torch.stack(inside_flag_list, dim=1)
        pts = pts.reshape(-1, num_point, 2)
        out_border_pts = pts[torch.where(inside_flag == 0)]

        if out_border_pts.size(0) > 0:
            corr_gt_boxes = gt_bboxes[torch.where(inside_flag == 0)[0]]
            corr_gt_boxes_center_x = (corr_gt_boxes[:, 0] +
                                      corr_gt_boxes[:, 4]) / 2.0
            corr_gt_boxes_center_y = (corr_gt_boxes[:, 1] +
                                      corr_gt_boxes[:, 5]) / 2.0
            corr_gt_boxes_center = torch.stack(
                [corr_gt_boxes_center_x, corr_gt_boxes_center_y], dim=1)
            distance_out_pts = 0.2 * ((
                (out_border_pts - corr_gt_boxes_center)**2).sum(dim=1).sqrt())
            loss = distance_out_pts.sum() / out_border_pts.size(0)

    return loss


def weighted_spatial_border_loss(pts, gt_bboxes, weight, avg_factor=None):
    """Weghted spatial border loss.

    Args:
        pts (torch.Tensor): point sets with shape (N, 9*2).
        gt_bboxes (torch.Tensor): gt_bboxes with polygon form with shape(N, 8)
        weight (torch.Tensor): weights for point sets with shape (N)

    Returns:
        loss (torch.Tensor)
    """

    weight = weight.unsqueeze(dim=1).repeat(1, 4)
    assert weight.dim() == 2
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = spatial_border_loss(pts, gt_bboxes)

    return torch.sum(loss)[None] / avg_factor
