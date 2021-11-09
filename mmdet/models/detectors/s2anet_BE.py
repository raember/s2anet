from .s2anet import S2ANetDetector
from ..registry import DETECTORS
from mmdet.core import bbox2result
import torch


@DETECTORS.register_module
class S2ANetDetectorBE(S2ANetDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANetDetectorBE, self).__init__(backbone, neck, bbox_head, train_cfg,
                                             test_cfg, pretrained)

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        
        # Implemented a loop through outputs of all ensemble members.
        # Returns all bbox_results and bbox_lists.
        # TODO: Pretty slow -> m-times slower, scales linearly.
        
        all_bbox_results = []
        all_bbox_lists = []
        
        indexes_to_select = torch.tensor([1, 3, 4])
        # 1: fam_bbox_pred, 3: odm_cls_score, 4: odm_bbox_pred

        m = outs[1][0].shape[0]  # TODO: add parameter m here!
        
        for k in torch.arange(0, m, device='cuda'):  # m = ensemble size
            outs_current_member = []
            for i in indexes_to_select:
                outs_current_index = []
                for j in torch.arange(0, 5):  # 5 = FPN depth
                    outs_current_index.append(
                        torch.index_select(outs[i][j], 0, k))
                outs_current_member.append(outs_current_index)

            outs_m = (outs[0],
                      outs_current_member[0],
                      outs[2],
                      outs_current_member[1],
                      outs_current_member[2])
            bbox_inputs = outs_m + (img_meta, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
            # one bbox_list is returned per ensemble member.
            all_bbox_lists.append(bbox_list)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            # bbox_results calculated per ensemble member.
            all_bbox_results.append(bbox_results[0])
        
        return all_bbox_results, all_bbox_lists
