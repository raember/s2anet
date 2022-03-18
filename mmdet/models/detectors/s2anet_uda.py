from .ssd_uda import SingleStageDetectorUDA
from ..registry import DETECTORS


@DETECTORS.register_module
class S2ANetDA(SingleStageDetectorUDA):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANetDA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                             test_cfg, pretrained)
