import json
import math

from mmdet.models.anchor_heads_rotated import S2ANetHead
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import (AnchorGeneratorRotated, anchor_target,
                        build_bbox_coder, delta2bbox_rotated, force_fp32,
                        images_to_levels, multi_apply, multiclass_nms_rotated)
from ...core.anchor.anchor_target import anchor_target_single

from ...ops import DeformConv
from ...ops.orn import ORConv2d, RotationInvariantPooling
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob


@HEADS.register_module
class S2ANetUDAHead(S2ANetHead):

    def __init__(self, *args,
                 loss_stat=dict(type='StatisticalLoss'),
                 nms={},
                 **kwargs):
        super(S2ANetUDAHead, self).__init__(*args, **kwargs)
        self.nms = nms
        loss_stat['cls_out_channels'] = self.cls_out_channels
        loss_stat['use_sigmoid_cls'] = self.use_sigmoid_cls
        loss_stat['target_means'] = self.target_means
        loss_stat['target_stds'] = self.target_stds
        self.loss_stat = build_loss(loss_stat)

    @force_fp32(apply_to=(
            'fam_cls_scores',
            'fam_bbox_preds',
            'odm_cls_scores',
            'odm_bbox_preds'))
    def loss(self,
             fam_cls_scores,
             fam_bbox_preds,
             refine_anchors,
             odm_cls_scores,
             odm_bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        if len(gt_bboxes[0]) != 0:
            result = super(S2ANetUDAHead, self).loss(
                fam_cls_scores,
                fam_bbox_preds,
                refine_anchors,
                odm_cls_scores,
                odm_bbox_preds,
                gt_bboxes,
                gt_labels,
                img_metas,
                cfg,
                gt_bboxes_ignore=gt_bboxes_ignore
            )
        else:
            result = {}
        # Make new loss
        propsals = []
        for ref_anch, odm_cls, odm_bbox in zip(refine_anchors, odm_cls_scores, odm_bbox_preds):
            lvl_propsals = self.get_bboxes(None, None, [ref_anch], [odm_cls], [odm_bbox], img_metas, self.nms)
            # In case it's empty, dtype long will be assumed
            bboxes = lvl_propsals[0][0].type(torch.float32)
            classes = lvl_propsals[0][1].type(torch.float32)
            for bboxs, clsses in lvl_propsals[1:]:
                bboxes = torch.cat((bboxes, bboxs.type(torch.float32)))
                classes = torch.cat((classes, clsses.type(torch.float32)))
            propsals.append((bboxes, classes))
        loss_stat_bbox, loss_stat_cls = multi_apply(self.loss_stat_single, propsals)

        if sum(loss_stat_cls) > 1E10 or \
           sum(loss_stat_bbox) > 1E10:
            print("bad loss")
        result.update(
            loss_stat_cls=loss_stat_cls,
            loss_stat_bbox=loss_stat_bbox,
        )
        return result

    def loss_stat_single(self, proposals):
        bboxes, classes = proposals
        valid = classes > 0
        bboxes, classes = bboxes[valid], classes[valid]
        if bboxes.shape[0] == 0:
            zero = torch.zeros((1,), device=bboxes.device)
            return zero, zero
        area = bboxes[:,3] * bboxes[:,2]
        angle = torch.fmod(bboxes[:,4] / math.pi * 180.0, 90.0)
        l1 = bboxes[:,2:4].min(dim=1).values
        l2 = bboxes[:,2:4].max(dim=1).values
        ratio = torch.addcdiv(torch.zeros_like(l1), 1, l2, l1)
        confid = bboxes[:,5]
        return self.loss_stat(area, angle, l1, l2, ratio, classes, confid)
