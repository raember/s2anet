import json

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
                 nms: dict,
                 loss_stat=dict(type='StatisticalLoss'),
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
        full = torch.empty((0, 7))
        bboxes: torch.Tensor
        classes: torch.Tensor
        for bboxes, classes in self.get_bboxes(None, None, refine_anchors, odm_cls_scores, odm_bbox_preds, img_metas, self.nms):
            block = torch.cat([bboxes, classes.reshape((len(classes), 1))], 1)
            full = torch.cat([full, block])
            areas = bboxes[:,3] * bboxes[:,2]
            angle = torch.rad2deg(bboxes[:,4])
            l1 = bboxes[:,2:4].min(dim=1)
            l2 = bboxes[:,2:4].max(dim=1)
            ratio = torch.div(l2, l1)
            torch.tens

        loss_stat_cls, loss_stat_bbox = multi_apply(
            self.loss_stat_single,
            odm_cls_scores,
            odm_bbox_preds,
            all_anchor_list,
            img_metas,
            cfg=cfg.odm_cfg)

        if sum(loss_stat_cls) > 1E10 or \
           sum(loss_stat_bbox) > 1E10:
            print("bad loss")
        result.update(
            loss_stat=loss_stat_cls,
            loss_stat_bbox=loss_stat_bbox,
        )
        return result

    def loss_stat_single(self, area, l1, l2, ratio, angle):
        # odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        # reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        # if reg_decoded_bbox:
        #     # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
        #     # is applied directly on the decoded bounding boxes, it
        #     # decodes the already encoded coordinates to absolute format.
        #     bbox_coder_cfg = cfg.get('bbox_coder', '')
        #     if bbox_coder_cfg == '':
        #         bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
        #     bbox_coder = build_bbox_coder(bbox_coder_cfg)
        #     anchors = anchors.reshape(-1, 5)
        #     odm_bbox_pred = bbox_coder.decode(anchors, odm_bbox_pred)
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_score]
        device = odm_cls_score[0].device
        refine_anchors = self.get_refine_anchors(featmap_sizes, anchors, img_metas, is_train=False, device=device)
        return self.loss_stat(odm_cls_score, odm_bbox_pred, refine_anchors, img_metas)
