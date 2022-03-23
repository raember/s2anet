import json
from abc import ABC
from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import LOSSES
from ...core import delta2bbox_rotated, multiclass_nms_rotated


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

    def forward(self, cls_scores: Tensor, bbox_preds: Tensor, refine_anchors: Tensor, img_metas: dict, **kwargs):
        num_levels = len(cls_scores)
        mlvl_bboxes = []
        mlvl_scores = []
        cls_score_list = [
            cls_scores[i].detach() for i in range(num_levels)
        ]
        bbox_pred_list = [
            bbox_preds[i].detach() for i in range(num_levels)
        ]
        img_shape = img_metas['img_shape']
        scale_factor = img_metas['scale_factor']
        mlvl_anchors = refine_anchors[0][0]
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # 135, 176, 124  vs  5, 176, 124
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the front when using sigmoid
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        return 0.0
