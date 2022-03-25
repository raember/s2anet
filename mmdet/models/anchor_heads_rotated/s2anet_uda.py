import json

from mmdet.models.anchor_heads_rotated import S2ANetHead
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import (AnchorGeneratorRotated, anchor_target,
                        build_bbox_coder, delta2bbox_rotated, force_fp32,
                        images_to_levels, multi_apply, multiclass_nms_rotated)

from ...ops import DeformConv
from ...ops.orn import ORConv2d, RotationInvariantPooling
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob


@HEADS.register_module
class S2ANetUDAHead(S2ANetHead):

    def __init__(self, *args,
                 stats_file: str = 'stats.json',
                 loss_stat_cls=dict(type='StatisticalClassLoss'),
                 loss_stat_bbox=dict(type='StatisticalBBoxLoss'),
                 **kwargs):
        super(S2ANetUDAHead, self).__init__(*args, **kwargs)
        self.stats = json.load(open(stats_file, mode='r'))
        loss_stat_cls['stats'] = self.stats
        loss_stat_bbox['stats'] = self.stats
        self.loss_stat_cls = build_loss(loss_stat_cls)
        self.loss_stat_bbox = build_loss(loss_stat_bbox)

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
            return super(S2ANetUDAHead, self).loss(
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
        # Make new loss
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        # check for size zero boxes
        for img_nr in range(len(gt_bboxes)):
            zero_inds = gt_bboxes[img_nr][:, 2:4] == 0
            gt_bboxes[img_nr][:, 2:4][zero_inds] = 1

        device = odm_cls_scores[0].device

        anchor_list, valid_flag_list = self.get_init_anchors(
            featmap_sizes, img_metas, device=device)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # Feature Alignment Module
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_fam_cls, losses_fam_bbox = multi_apply(
            self.loss_fam_single,
            fam_cls_scores,
            fam_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.fam_cfg)

        # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, device=device)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0)
                             for anchors in refine_anchors_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(torch.cat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.odm_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_odm_cls, losses_odm_bbox = multi_apply(
            self.loss_odm_single,
            odm_cls_scores,
            odm_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.odm_cfg)

        self.last_vals = dict(
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            fam_cls_scores=fam_cls_scores,
            fam_bbox_preds=fam_bbox_preds,
            refine_anchors=refine_anchors,
            odm_cls_scores=odm_cls_scores,
            odm_bbox_preds=odm_bbox_preds,
        )
        if sum(losses_fam_cls) > 1E10 or \
           sum(losses_fam_bbox) > 1E10 or \
           sum(losses_odm_cls) > 1E10 or \
           sum(losses_odm_bbox) > 1E10:
            print("bad loss")
        return dict(loss_fam_cls=losses_fam_cls,
                    loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls,
                    loss_odm_bbox=losses_odm_bbox)
