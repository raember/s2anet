"""DEEPSCORESV2

Provides access to the DEEPSCORESV2 database with a COCO-like interface. The
only changes made compared to the coco.py file are the class labels.

Author:
    Lukas Tuggener <tugg@zhaw.ch>
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    November 23, 2019
"""
import json
from collections import defaultdict
from typing import Dict, Tuple

import mmcv
from obb_anns import OBBAnns

from .coco import *
from .deepscoresv2 import DeepScoresV2Dataset

@DATASETS.register_module
class DeepScoresV2Dataset_Hybrid(DeepScoresV2Dataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 use_oriented_bboxes=True):
        super(DeepScoresV2Dataset_Hybrid, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                                         img_prefix=img_prefix, seg_prefix=seg_prefix, proposal_file=proposal_file,
                                                         test_mode=test_mode, filter_empty_gt=filter_empty_gt, use_oriented_bboxes=use_oriented_bboxes)
        #self.CLASSES = self.get_classes(classes)

    def _parse_ann_info(self, img_info, ann_info):
        img_info, ann_info = img_info[0], ann_info[0]
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = np.zeros((0, 8 if self.use_oriented_bboxes else 4), dtype=np.float32)

        for i, ann in ann_info.iterrows():
            # we have no ignore feature
            if ann['area'] <= 0:
                continue

            if self.use_oriented_bboxes:
                if ann["cat_id"][0] in [122,123,124,134]:
                # for tie,slur,beam,tuplet_bracket use true o_bbox
                    bbox = ann['o_bbox']
                else:
                # use a_bbox in o_bbox format
                    bbox = [ann['a_bbox'][x] for x in [2, 3, 2, 1, 0, 1, 0, 3]]
            else:
                bbox = ann['a_bbox']
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['cat_id'][0]])

        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=None,
            seg_map=None)
        return ann


