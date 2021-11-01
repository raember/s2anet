from __future__ import division

import numpy as np
import math
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
from .s2anet_head import S2ANetHead, AlignConv, bbox_decode

# Visualization imports
import debugging.visualization_tools as vt
from mmcv.visualization import imshow_det_bboxes
from mmcv.image import tensor2imgs

@HEADS.register_module
class S2ANetHeadBE(S2ANetHead):

    def __init__(self, **kwargs):
        super(S2ANetHeadBE, self).__init__(**kwargs)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.fam_reg_convs = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.fam_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        # TODO test this:
        self.fam_reg = Ensemble_Conv2d(self.feat_channels, self.num_anchors * 5, 1)
        self.fam_cls = Ensemble_Conv2d(self.feat_channels, self.num_anchors *self.cls_out_channels, 1)

        self.align_conv = AlignConv(self.feat_channels, self.feat_channels, kernel_size=3, anchor_num=self.num_anchors)

        if self.with_orconv:
            self.or_conv = ORConv2d(self.feat_channels, int(
                self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        self.or_pool = RotationInvariantPooling(256, 8)

        self.odm_reg_convs = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels /
                      8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.odm_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
        # TODO: test this
        self.odm_cls = Ensemble_Conv2d(
            self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.odm_reg = Ensemble_Conv2d(self.feat_channels, self.num_anchors *5, 3, padding=1)

    def init_weights(self):
        # Initializes all weights except those of the Ensemble_Conv2d() layers
        for m in self.fam_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.fam_cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        
        # Ensemble_Conv2d() layer weights are initialized within layer
        # normal_init(self.fam_reg, std=0.01)
        # normal_init(self.fam_cls, std=0.01, bias=bias_cls)

        self.align_conv.init_weights()

        normal_init(self.or_conv, std=0.01)
        for m in self.odm_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.odm_cls_convs:
            normal_init(m.conv, std=0.01)

        # Ensemble_Conv2d() layer weights are initialized within layer
        # normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        # normal_init(self.odm_reg, std=0.01)

    def forward_single(self, x, stride):
        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat)
        fam_bbox_pred = self.fam_reg(fam_reg_feat)  # Implemented BatchEnsemble here

        # only forward during training
        if self.training:
            fam_cls_feat = x
            for fam_cls_conv in self.fam_cls_convs:
                fam_cls_feat = fam_cls_conv(fam_cls_feat)
            fam_cls_score = self.fam_cls(fam_cls_feat)  # Implemented BatchEnsemble here
        else:
            fam_cls_score = None

        num_level = self.anchor_strides.index(stride)
        featmap_size = fam_bbox_pred.shape[-2:]
        if (num_level, featmap_size) in self.base_anchors:
            init_anchors = self.base_anchors[(num_level, featmap_size)]
        else:
            device = fam_bbox_pred.device
            init_anchors = self.anchor_generators[num_level].grid_anchors(
                featmap_size, self.anchor_strides[num_level], device=device)
            self.base_anchors[(num_level, featmap_size)] = init_anchors

        # Problem: fam_bbox_pred has ensemble-shape! -> could take 1st dim (i.e. 1st member) only...
        fam_bbox_pred_1d = fam_bbox_pred.detach()
        fam_bbox_pred_1d = fam_bbox_pred_1d[0:1, :, :, :]
        refine_anchor = bbox_decode(
            fam_bbox_pred_1d,
            init_anchors,
            self.target_means,
            self.target_stds,
            self.num_anchors)

        align_feat = self.align_conv(x, refine_anchor.clone(), stride)

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat

        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        odm_cls_score = self.odm_cls(odm_cls_feat)  # Implemented BatchEnsemble here
        odm_bbox_pred = self.odm_reg(odm_reg_feat)  # Implemented BatchEnsemble here

        # All outputs have ensemble-compatible shape except refine_anchor
        return fam_cls_score, fam_bbox_pred, refine_anchor, odm_cls_score, odm_bbox_pred
    #
    # def forward(self, feats):
    #     return multi_apply(self.forward_single, feats, self.anchor_strides)
    #
    # def get_init_anchors(self,
    #                      featmap_sizes,
    #                      img_metas,
    #                      device='cuda'):
    #     """Get anchors according to feature map sizes.
    #
    #     Args:
    #         featmap_sizes (list[tuple]): Multi-level feature map sizes.
    #         img_metas (list[dict]): Image meta info.
    #         device (torch.device | str): device for returned tensors
    #
    #     Returns:
    #         tuple: anchors of each image, valid flags of each image
    #     """
    #     num_imgs = len(img_metas)
    #     num_levels = len(featmap_sizes)
    #
    #     # since feature map sizes of all images are the same, we only compute
    #     # anchors for one time
    #     multi_level_anchors = []
    #     for i in range(num_levels):
    #         anchors = self.anchor_generators[i].grid_anchors(
    #             featmap_sizes[i], self.anchor_strides[i], device=device)
    #         multi_level_anchors.append(anchors)
    #     anchor_list = [multi_level_anchors for _ in range(num_imgs)]
    #
    #     # for each image, we compute valid flags of multi level anchors
    #     valid_flag_list = []
    #     for img_id, img_meta in enumerate(img_metas):
    #         multi_level_flags = []
    #         for i in range(num_levels):
    #             anchor_stride = self.anchor_strides[i]
    #             feat_h, feat_w = featmap_sizes[i]
    #             h, w, _ = img_meta['pad_shape']
    #             valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
    #             valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
    #             flags = self.anchor_generators[i].valid_flags(
    #                 (feat_h, feat_w), (valid_feat_h, valid_feat_w),
    #                 device=device)
    #             multi_level_flags.append(flags)
    #         valid_flag_list.append(multi_level_flags)
    #     return anchor_list, valid_flag_list
    #
    # def get_refine_anchors(self,
    #                        featmap_sizes,
    #                        refine_anchors,
    #                        img_metas,
    #                        is_train=True,
    #                        device='cuda'):
    #     num_levels = len(featmap_sizes)
    #
    #     refine_anchors_list = []
    #     for img_id, img_meta in enumerate(img_metas):
    #         mlvl_refine_anchors = []
    #         for i in range(num_levels):
    #             refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
    #             mlvl_refine_anchors.append(refine_anchor)
    #         refine_anchors_list.append(mlvl_refine_anchors)
    #
    #     valid_flag_list = []
    #     if is_train:
    #         for img_id, img_meta in enumerate(img_metas):
    #             multi_level_flags = []
    #             for i in range(num_levels):
    #                 anchor_stride = self.anchor_strides[i]
    #                 feat_h, feat_w = featmap_sizes[i]
    #                 h, w, _ = img_meta['pad_shape']
    #                 valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
    #                 valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
    #                 flags = self.anchor_generators[i].valid_flags(
    #                     (feat_h, feat_w), (valid_feat_h, valid_feat_w),
    #                     device=device)
    #                 multi_level_flags.append(flags)
    #             valid_flag_list.append(multi_level_flags)
    #     return refine_anchors_list, valid_flag_list
    #
    # @force_fp32(apply_to=(
    #     'fam_cls_scores',
    #     'fam_bbox_preds',
    #     'odm_cls_scores',
    #     'odm_bbox_preds'))
    
    # TODO make loss ensemble-compatible
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
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.fam_cfg,
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

        m = fam_cls_scores[0].shape[0]
        losses_fam_cls_list = []
        losses_fam_bbox_list = []
        for i in range(m): # TODO make it a parameter m
            fam_cls_scores_tmp = []
            for j in range(len(fam_cls_scores)):
                fam_cls_scores_tmp.append(odm_cls_scores[j][i:i+1, :, :, :])

            fam_bbox_preds_tmp = []
            for j in range(len(fam_bbox_preds)):
                fam_bbox_preds_tmp.append(odm_bbox_preds[j][i:i+1, :, :, :])
                
            losses_fam_cls, losses_fam_bbox = multi_apply(
                self.loss_fam_single,
                fam_cls_scores_tmp,
                fam_bbox_preds_tmp,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples,
                cfg=cfg.fam_cfg)
            losses_fam_cls_list.append(losses_fam_cls)
            losses_fam_bbox_list.append(losses_fam_bbox)

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

        losses_odm_cls_list = []
        losses_odm_bbox_list = []
        for i in range(m): # TODO make it a parameter m
            odm_cls_scores_tmp = []
            for j in range(len(odm_cls_scores)):
                odm_cls_scores_tmp.append(odm_cls_scores[j][i:i+1, :, :, :])

            odm_bbox_preds_tmp = []
            for j in range(len(odm_bbox_preds)):
                odm_bbox_preds_tmp.append(odm_bbox_preds[j][i:i+1, :, :, :])
                
            losses_odm_cls, losses_odm_bbox = multi_apply(
                self.loss_odm_single,
                odm_cls_scores_tmp,
                odm_bbox_preds_tmp,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples,
                cfg=cfg.odm_cfg)
            losses_odm_cls_list.append(losses_odm_cls)
            losses_odm_bbox_list.append(losses_odm_bbox)


        # Clumsy loss-mean
        
        losses_odm_cls = []
        for i in range(len(losses_odm_cls_list[0])):
            i_index_sum = 0
            for j in range(len(losses_odm_cls_list)):
                i_index_sum += losses_odm_cls_list[j][i]
            losses_odm_cls.append(i_index_sum/len(losses_odm_cls_list))

        #losses_odm_cls = list(torch.tensor(losses_odm_cls_list, requires_grad=True, device='cuda').mean(dim=0))

        losses_odm_bbox = []
        for i in range(len(losses_odm_bbox_list[0])):
            i_index_sum = 0
            for j in range(len(losses_odm_bbox_list)):
                i_index_sum += losses_odm_bbox_list[j][i]
            losses_odm_bbox.append(i_index_sum/len(losses_odm_bbox_list))

        #losses_odm_bbox = list(torch.tensor(losses_odm_bbox_list, requires_grad=True, device='cuda').mean(dim=0))

        losses_fam_cls = []
        for i in range(len(losses_fam_cls_list[0])):
            i_index_sum = 0
            for j in range(len(losses_fam_cls_list)):
                i_index_sum += losses_fam_cls_list[j][i]
            losses_fam_cls.append(i_index_sum/len(losses_fam_cls_list))

        #losses_fam_cls = list(torch.tensor(losses_fam_cls_list, requires_grad=True, device='cuda').mean(dim=0))

        losses_fam_bbox = []
        for i in range(len(losses_fam_bbox_list[0])):
            i_index_sum = 0
            for j in range(len(losses_fam_bbox_list)):
                i_index_sum += losses_fam_bbox_list[j][i]
            losses_fam_bbox.append(i_index_sum/len(losses_fam_bbox_list))

        #losses_fam_bbox = list(torch.tensor(losses_fam_bbox_list, requires_grad=True, device='cuda').mean(dim=0))


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

    def loss_fam_single(self,
                        fam_cls_score,
                        fam_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        fam_cls_score = fam_cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_fam_cls = self.loss_fam_cls(
            fam_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_bbox_coder(bbox_coder_cfg)
            anchors = anchors.reshape(-1, 5)
            fam_bbox_pred = bbox_coder.decode(anchors, fam_bbox_pred)
        loss_fam_bbox = self.loss_fam_bbox(
            fam_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_fam_cls, loss_fam_bbox

    def loss_odm_single(self,
                        odm_cls_score,
                        odm_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        odm_cls_score = odm_cls_score.permute(0, 2, 3,
                                              1).reshape(-1, self.cls_out_channels)
        loss_odm_cls = self.loss_odm_cls(
            odm_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_bbox_coder(bbox_coder_cfg)
            anchors = anchors.reshape(-1, 5)
            odm_bbox_pred = bbox_coder.decode(anchors, odm_bbox_pred)
        loss_odm_bbox = self.loss_odm_bbox(
            odm_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_odm_cls, loss_odm_bbox
    #
    # @force_fp32(apply_to=(
    #     'fam_cls_scores',
    #     'fam_bbox_preds',
    #     'odm_cls_scores',
    #     'odm_bbox_preds'))
    # def get_bboxes(self,
    #                fam_cls_scores,
    #                fam_bbox_preds,
    #                refine_anchors,
    #                odm_cls_scores,
    #                odm_bbox_preds,
    #                img_metas,
    #                cfg,
    #                rescale=False):
    #     assert len(odm_cls_scores) == len(odm_bbox_preds)
    #
    #     featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
    #     num_levels = len(odm_cls_scores)
    #     device = odm_cls_scores[0].device
    #
    #     refine_anchors = self.get_refine_anchors(
    #         featmap_sizes, refine_anchors, img_metas, is_train=False, device=device)
    #     result_list = []
    #     for img_id in range(len(img_metas)):
    #         cls_score_list = [
    #             odm_cls_scores[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         bbox_pred_list = [
    #             odm_bbox_preds[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         img_shape = img_metas[img_id]['img_shape']
    #         scale_factor = img_metas[img_id]['scale_factor']
    #         proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
    #                                            refine_anchors[0][0], img_shape,
    #                                            scale_factor, cfg, rescale)
    #
    #         result_list.append(proposals)
    #     return result_list
    #
    # def get_bboxes_single(self,
    #                       cls_score_list,
    #                       bbox_pred_list,
    #                       mlvl_anchors,
    #                       img_shape,
    #                       scale_factor,
    #                       cfg,
    #                       rescale=False):
    #     """
    #     Transform outputs for a single batch item into labeled boxes.
    #     """
    #     assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     for cls_score, bbox_pred, anchors in zip(cls_score_list,
    #                                              bbox_pred_list, mlvl_anchors):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         cls_score = cls_score.permute(
    #             1, 2, 0).reshape(-1, self.cls_out_channels)
    #
    #         if self.use_sigmoid_cls:
    #             scores = cls_score.sigmoid()
    #         else:
    #             scores = cls_score.softmax(-1)
    #
    #         bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
    #         # anchors = rect2rbox(anchors)
    #         nms_pre = cfg.get('nms_pre', -1)
    #         if nms_pre > 0 and scores.shape[0] > nms_pre:
    #             # Get maximum scores for foreground classes.
    #             if self.use_sigmoid_cls:
    #                 max_scores, _ = scores.max(dim=1)
    #             else:
    #                 max_scores, _ = scores[:, 1:].max(dim=1)
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             anchors = anchors[topk_inds, :]
    #             bbox_pred = bbox_pred[topk_inds, :]
    #             scores = scores[topk_inds, :]
    #         bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
    #                                     self.target_stds, img_shape)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #     mlvl_bboxes = torch.cat(mlvl_bboxes)
    #     if rescale:
    #         mlvl_bboxes[..., :4] /= mlvl_bboxes.new_tensor(scale_factor)
    #     mlvl_scores = torch.cat(mlvl_scores)
    #     if self.use_sigmoid_cls:
    #         # Add a dummy background class to the front when using sigmoid
    #         padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    #         mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
    #     det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes,
    #                                                     mlvl_scores,
    #                                                     cfg.score_thr, cfg.nms,
    #                                                     cfg.max_per_img)
    #     return det_bboxes, det_labels
    #
    # def get_visualization(self, input_img, classes, test_cfg):
    #     stitched = "nonew3"
    #     batch_size = input_img.shape[0]
    #     img = tensor2imgs(input_img, **self.last_vals['img_metas'][0]['img_norm_cfg'])[0] #get input image
    #     from PIL import Image
    #     #Image.fromarray(img).show()
    #     from mmdet.core import rotated_box_to_poly_np
    #     gt = rotated_box_to_poly_np(self.last_vals['gt_bboxes'][0].cpu().numpy())
    #
    #     img_gt = imshow_det_bboxes(img.copy(),gt,
    #                                     self.last_vals['gt_labels'][0].cpu().numpy(),
    #                                     class_names=classes, show=False, show_label=True, rotated=True)
    #
    #     det_boxes_labels = self.get_bboxes(fam_cls_scores=self.last_vals['fam_cls_scores'],
    #                fam_bbox_preds=self.last_vals['fam_bbox_preds'],
    #                refine_anchors=self.last_vals['refine_anchors'],
    #                odm_cls_scores=self.last_vals['odm_cls_scores'],
    #                odm_bbox_preds=self.last_vals['odm_bbox_preds'],
    #                img_metas=self.last_vals['img_metas'],
    #                cfg=test_cfg)[0]
    #     det_boxes = rotated_box_to_poly_np(det_boxes_labels[0].cpu().numpy())
    #     det_labels = det_boxes_labels[1].cpu().numpy()
    #     if len(det_boxes)>0:
    #         img_det = imshow_det_bboxes(img.copy(), det_boxes,
    #                                         det_labels.astype(np.int)+1,
    #                                         class_names=classes, show=False, show_label=True, rotated=True)
    #     else:
    #         img_det = img.copy()
    #     stitched = vt.stitch_big_image([[img_gt],
    #                                     [img_det]])
    #
    #     return [{"name": "stitched_img", "image": stitched}]
        # Image.fromarray(stitched).show()
# def bbox_decode(
#         bbox_preds,
#         anchors,
#         means=[0, 0, 0, 0, 0],
#         stds=[1, 1, 1, 1, 1],
#         num_anchors=1):
#     """
#     Decode bboxes from deltas
#     :param bbox_preds: [N,5,H,W]
#     :param anchors: [H*W,5]
#     :param means: mean value to decode bbox
#     :param stds: std value to decode bbox
#     :return: [N,H,W,5]
#     """
#     num_imgs, _, H, W = bbox_preds.shape
#     bboxes_list = []
#     for img_id in range(num_imgs):
#         bbox_pred = bbox_preds[img_id]
#         # bbox_pred.shape=[5,H,W]
#         bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
#         bboxes = delta2bbox_rotated(
#             anchors, bbox_delta, means, stds, wh_ratio_clip=1e-6)
#         bboxes = bboxes.reshape(num_anchors,H, W, 5)
#         bboxes_list.append(bboxes)
#     return torch.stack(bboxes_list, dim=0)


# class AlignConv(nn.Module):
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  deformable_groups=1,
#                  anchor_num = 1):
#         super(AlignConv, self).__init__()
#         self.kernel_size = kernel_size
#         self.deform_conv = DeformConv(in_channels,
#                                       out_channels,
#                                       kernel_size=kernel_size,
#                                       padding=(kernel_size - 1) // 2,
#                                       deformable_groups=deformable_groups)
#         self.relu = nn.ReLU(inplace=True)
#         self.anchor_num = anchor_num
#
#     def init_weights(self):
#         normal_init(self.deform_conv, std=0.01)
#
#     @torch.no_grad()
#     def get_offset(self, anchors, featmap_size, stride):
#         dtype, device = anchors.dtype, anchors.device
#         feat_h, feat_w = featmap_size
#         pad = (self.kernel_size - 1) // 2
#         idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
#         yy, xx = torch.meshgrid(idx, idx)
#         xx = xx.reshape(-1)
#         yy = yy.reshape(-1)
#
#         # get sampling locations of default conv
#         xc = torch.arange(0, feat_w, device=device, dtype=dtype)
#         yc = torch.arange(0, feat_h, device=device, dtype=dtype)
#         yc, xc = torch.meshgrid(yc, xc)
#         xc = xc.reshape(-1)
#         yc = yc.reshape(-1)
#         x_conv = xc[:, None] + xx
#         y_conv = yc[:, None] + yy
#
#         # get sampling locations of anchors
#         x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
#         x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
#         cos, sin = torch.cos(a), torch.sin(a)
#         dw, dh = w / self.kernel_size, h / self.kernel_size
#         x, y = dw[:, None] * xx, dh[:, None] * yy
#         xr = cos[:, None] * x - sin[:, None] * y
#         yr = sin[:, None] * x + cos[:, None] * y
#         x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
#         # get offset filed
#         offset_x = x_anchor - x_conv
#         offset_y = y_anchor - y_conv
#         # x, y in anchors is opposite in image coordinates,
#         # so we stack them with y, x other than x, y
#         offset = torch.stack([offset_y, offset_x], dim=-1)
#         # NA,ks*ks*2
#         offset = offset.reshape(anchors.size(0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
#         return offset
#
#     # def forward(self, x, anchors, stride):
#     #     num_imgs, H, W = anchors.shape[:3]
#     #     offset_list = [
#     #         self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
#     #         for i in range(num_imgs)
#     #     ]
#     #     offset_tensor = torch.stack(offset_list, dim=0)
#     #     x = self.relu(self.deform_conv(x, offset_tensor))
#     #     return x
#
#     def forward(self, x, anchors, stride):
#         num_imgs = anchors.shape[0]
#         H, W = anchors.shape[2:4]
#         out = list()
#         for i in range(self.anchor_num):
#             offset_list = [self.get_offset(anchors[ii,i,...].reshape(-1,5), (H,W), stride)
#                            for ii in range(num_imgs)]
#             offset_tensor = torch.stack(offset_list, dim=0)
#             out.append(self.relu(self.deform_conv(x, offset_tensor)))
#         out = torch.stack(out, dim=1)
#         out, _ = torch.max(out, dim=1, keepdim=False)
#         return out


# Class from LP_BNN repo
class Ensemble_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, first_layer=False, num_models=30, train_gamma=True,
                 bias=True, constant_init=False, p=0.5, random_sign_init=False):
        super(Ensemble_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels))
        self.train_gamma = train_gamma
        self.random_sign_init = random_sign_init
        self.constant_init = constant_init
        self.probability = p
        if train_gamma:
            self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        self.num_models = num_models
        if bias:
            #self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.first_layer = first_layer

    def reset_parameters(self):
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            if self.random_sign_init:
                if self.probability  == -1:
                    with torch.no_grad():
                        factor = torch.ones(
                            self.num_models, device=self.alpha.device).bernoulli_(0.5)
                        factor.mul_(2).add_(-1)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma.fill_(1.)
                            self.gamma.data = (self.gamma.t() * factor).t()
                elif self.probability  == -2:
                    with torch.no_grad():
                        positives_num = self.num_models // 2
                        factor1 = torch.Tensor([1 for i in range(positives_num)])
                        factor2 = torch.Tensor(
                            [-1 for i in range(self.num_models-positives_num)])
                        factor = torch.cat([factor1, factor2]).to(self.alpha.device)
                        self.alpha.data = (self.alpha.t() * factor).t()
                        if self.train_gamma:
                            self.gamma.fill_(1.)
                            self.gamma.data = (self.gamma.t() * factor).t()
                else:
                    with torch.no_grad():
                        self.alpha.bernoulli_(self.probability)
                        self.alpha.mul_(2).add_(-1)
                        if self.train_gamma:
                            self.gamma.bernoulli_(self.probability)
                            self.gamma.mul_(2).add_(-1)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            #nn.init.normal_(self.alpha, mean=1., std=1)
            if self.train_gamma:
                nn.init.normal_(self.gamma, mean=1., std=0.5)
                #nn.init.normal_(self.gamma, mean=1., std=1)
            if self.random_sign_init:
                with torch.no_grad():
                    alpha_coeff = torch.randint_like(self.alpha, low=0, high=2)
                    alpha_coeff.mul_(2).add_(-1)
                    self.alpha *= alpha_coeff
                    if self.train_gamma:
                        gamma_coeff = torch.randint_like(self.gamma, low=0, high=2)
                        gamma_coeff.mul_(2).add_(-1)
                        self.gamma *= gamma_coeff
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_indices(self, indices):
        self.indices = indices

    def forward(self, x):
        # Tile here thus making input ensemble-compatible.
        x = tile(x, 0, self.num_models)
        if not self.training and self.first_layer:
            # Repeated pattern in test: [[A,B,C],[A,B,C]]
            x = torch.cat([x for i in range(self.num_models)], dim=0)
        if self.train_gamma:
            num_examples_per_model = int(x.size(0) / self.num_models)
            extra = x.size(0) - (num_examples_per_model * self.num_models)

            # Repeated pattern: [[A,A],[B,B],[C,C]]
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)
            gamma = torch.cat(
                [self.gamma for i in range(num_examples_per_model)],
                dim=1).view([-1, self.out_channels])
            gamma.unsqueeze_(-1).unsqueeze_(-1)
            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)

            if extra != 0:
                alpha = torch.cat([alpha, alpha[:extra]], dim=0)
                gamma = torch.cat([gamma, gamma[:extra]], dim=0)
                if self.bias is not None:
                    bias = torch.cat([bias, bias[:extra]], dim=0)
            result = self.conv(x*alpha)*gamma

            #import time
            #start_time = time.time()
            #print('reshape takes {} s'.format(time.time()-start_time))
            #start_time = time.time()
            #print('conv and dot product takes {} s'.format(time.time()-start_time))
            #alpha = self.alpha.repeat(1, num_examples_per_model).view(-1, self.in_channels)
            #alpha.unsqueeze_(-1).unsqueeze_(-1)
            #gamma = self.gamma.repeat(1, num_examples_per_model) .view(-1, self.out_channels)
            #gamma.unsqueeze_(-1).unsqueeze_(-1)
            #if self.bias is not None:
            #    bias = self.bias.repeat(1, num_examples_per_model) .view(-1, self.out_channels)
            #    bias.unsqueeze_(-1).unsqueeze_(-1)
            #result = self.conv(x*alpha)*gamma
            #tmp = self.conv(x*torch.randn_like(x))
            #result = torch.randn_like(tmp) * tmp

            return result + bias if self.bias is not None else result
        else:
            num_examples_per_model = int(x.size(0) / self.num_models)
            # Repeated pattern: [[A,A],[B,B],[C,C]]
            alpha = torch.cat(
                [self.alpha for i in range(num_examples_per_model)],
                dim=1).view([-1, self.in_channels])
            alpha.unsqueeze_(-1).unsqueeze_(-1)

            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)],
                    dim=1).view([-1, self.out_channels])
                bias.unsqueeze_(-1).unsqueeze_(-1)
            result = self.conv(x*alpha)
            return result + bias if self.bias is not None else result
        

# utility function to reshape batch ensemble layer inputs and targets
# function from LP_BNN repo
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)