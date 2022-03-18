import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetectorUDA(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetectorUDA, self).__init__()
        self.backbone_src = builder.build_backbone(backbone)
        self.backbone_tgt = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.last_vals = None

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorUDA, self).init_weights(pretrained)
        self.backbone_src.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, src=True):
        """Directly extract features from the backbone+neck
        """
        if src:
            x = self.backbone_src(img)
        else:
            x = self.backbone_tgt(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      src_img,
                      tgt_img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        #Source
        x_src = self.extract_feat(src_img)
        outs_src = self.bbox_head(x_src)
        loss_inputs = outs_src + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        self.last_vals = dict(img=src_img)

        #Target
        x_tgt = self.extract_feat(tgt_img, src=False)
        outs_tgt = self.bbox_head(x_tgt) #DO WE NEED THIS?
        return losses, x_src, x_tgt
        # return losses, outs_src, outs_tgt


    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError