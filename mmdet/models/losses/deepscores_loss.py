import json
from abc import ABC
from pathlib import Path
from typing import Union, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import LOSSES
from ...datasets.deepscoresv2 import get_thresholds


@LOSSES.register_module
class StatisticalLoss(nn.Module):

    def __init__(self, stats_file: str, cls_out_channels: int, use_sigmoid_cls: bool,
                 target_means: List[int], target_stds: List[int]):
        super().__init__()
        self.stats = json.load(open(stats_file, mode='r'))
        self.cls_out_channels = cls_out_channels
        self.use_sigmoid_cls = use_sigmoid_cls
        self.target_means = target_means
        self.target_stds = target_stds

    def forward(self, area: Tensor, angle: Tensor, l1: Tensor, l2: Tensor, ratio: Tensor, cls: Tensor):
        loss_bbox = 0.0
        losses = torch.zeros((area.shape[0], 0), device=cls.device)
        for val, (val_min, val_max, val_std) in zip((area, angle, l1, l2, ratio), self.get_thresholds_by_classes(cls)):
            mean = torch.mean(torch.cat((val_min.reshape((val.shape[0], 1)), val_max.reshape((val.shape[0], 1))), dim=1), dim=1)
            zeros = torch.zeros_like(val, device=val.device)
            value = zeros.addcdiv(torch.abs(val - mean) - (val_max - mean), val_std)
            # Anything above 0 is outside a threshold and already scaled for loss
            losses = torch.cat((losses, value.where(value > 0.0, zeros).reshape((val.shape[0], 1))), dim=1)
        loss_bbox = losses.mean()
        #TODO: Calculate loss for class
        loss_cls = torch.zeros((1,), device=cls.device)
        return loss_bbox, loss_cls

    def get_thresholds_by_classes(self, cls: Tensor) -> Tuple[
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor]]:
        area_min, area_max, area_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        angle_min, angle_max, angle_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        l1_min, l1_max, l1_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        l2_min, l2_max, l2_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)
        ratio_min, ratio_max, ratio_std = torch.zeros_like(cls, device=cls.device), torch.full_like(cls, 2 ** 24, device=cls.device), torch.ones_like(cls, device=cls.device)

        all_threshold_tensors = (area_min, area_max, area_std), (angle_min, angle_max, angle_std),\
                                (l1_min, l1_max, l1_std), (l2_min, l2_max, l2_std), (ratio_min, ratio_max, ratio_std)
        for i, cls_id in enumerate(cls.tolist()):
            cls_id = int(cls_id) + 2
            cls_stat = self.stats.get(str(cls_id), {'id': cls_id})
            threshs = get_thresholds(cls_stat)
            for (key, (low, high)), (val_min, val_max, val_std) in zip(threshs.items(), all_threshold_tensors):
                if low is None or high is None:
                    continue
                val_min[i] = low
                val_max[i] = high
                val_std[i] = max(1.0, cls_stat[key]['std'])
        return all_threshold_tensors
