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


@DATASETS.register_module
class DeepScoresV2Dataset(CocoDataset):

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
        self.filter_empty_gt = filter_empty_gt
        super(DeepScoresV2Dataset, self).__init__(ann_file, pipeline, data_root, img_prefix, seg_prefix, proposal_file, test_mode)
        #self.CLASSES = self.get_classes(classes)
        self.use_oriented_bboxes = use_oriented_bboxes

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names


    def load_annotations(self, ann_file):
        self.obb = OBBAnns(ann_file)
        self.obb.load_annotations()
        self.obb.set_annotation_set_filter(['deepscores'])
        # self.obb.set_class_blacklist(["staff"])
        self.cat_ids = list(self.obb.get_cats().keys())
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.label2cat = {v: k for k, v in self.cat2label.items()}
        self.CLASSES = tuple([v["name"] for (k, v) in self.obb.get_cats().items()])
        self.img_ids = [id['id'] for id in self.obb.img_info]

        return self.obb.img_info

    def get_ann_info(self, idx):
        return self._parse_ann_info(*self.obb.get_img_ann_pair(idxs=[idx]))

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.obb.img_info):
            if self.filter_empty_gt and len(img_info['ann_ids']) == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        img_info, ann_info = img_info[0], ann_info[0]
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = np.zeros((0, 8 if self.use_oriented_bboxes else 4), dtype=np.float32)

        for i, ann in ann_info.iterrows():
            # we have no ignore feature
            if ann['area'] <= 0:
                continue

            bbox = ann['o_bbox' if self.use_oriented_bboxes else 'a_bbox']
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

    def prepare_json_dict(self, results):
        json_results = {"annotation_set": "deepscores", "proposals": []}
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['img_id'] = img_id

                    if len(bboxes[i]) == 8:
                        data['bbox'] = [str(nr) for nr in bboxes[i]]
                        data['score'] = 1
                    else:
                        data['bbox'] = [str(nr) for nr in bboxes[i][0:-1]]
                        data['score'] = str(bboxes[i][-1])
                    data['cat_id'] = self.label2cat[label]
                    json_results["proposals"].append(data)
        return json_results

    def write_results_json(self, results, filename=None):
        if filename is None:
            filename = "deepscores_results.json"
        json_results = self.prepare_json_dict(results)

        with open(filename, "w") as fo:
            json.dump(json_results, fo)

        return filename

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 average_thrs=False,
                 work_dir = None):
        """Evaluation in COCO protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str: float]
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        filename = self.write_results_json(results)

        self.obb.load_proposals(filename)
        metric_results = self.obb.calculate_metrics(iou_thrs=iou_thrs, classwise=classwise, average_thrs=average_thrs)

        categories = self.obb.get_cats()
        metric_results = {categories[key]['name']: value for (key, value) in metric_results.items()}

        # add occurences
        occurences_by_class = self.obb.get_class_occurences()
        for (key, value) in metric_results.items():
            value.update(no_occurences=occurences_by_class[key])

        if work_dir is not None:
            import pickle
            import os
            out_file = os.path.join(work_dir, "dsv2_metrics.pkl")
            pickle.dump(metric_results, open(out_file, 'wb'))
            
            if self.data_root is None:
                self.data_root = '/'.join(self.ann_file.split('/')[0:2]) + '/'

            out_dir = os.path.join(work_dir, "visualized_proposals/")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            for img_info in self.obb.img_info:
                self.obb.visualize(img_id=img_info['id'],
                                   data_root=self.data_root,
                                   out_dir=out_dir
                                   )

        print(metric_results)
        return metric_results

area_thr_def = (1, 1.0)
angle_thr_def = (85, 5)  # Between 5 and 85°
l1_thr_def = (1, 1.0)
l2_thr_def = (1, 1.0)
ratio_thr_def = (1, 2)
threshold_classes = ['area', 'angle', 'l1', 'l2', 'edge-ratio']
thresholds = {
    # tuple: Upper and lower bound. vs single value: symmetric bounds
    # int/float: absolute threshold
    # If high value first: Inverted threshold (mainly used for angles)
    #   Area,           Angle,          L1,         L2,         Ratio           # Class name
    1:  ((1000, 30000), (85, 5),        (5, 50),    (10, 600),  (7, 50)),       # brace
    2:  ((23, 258),     (90, 0),        (1, 5),     (17, 100),  (5, 100)),      # ledgerLine
    3:  ((20, 90),      None,           (4, 10),    (4, 13),    (1, 1.7)),      # repeatDot
    4:  ((1500, 9000),  (85, 5),        (25, 90),   (30, 110),  (1, 1.7)),      # segno
    5:  ((1000, 9000),  None,           (25, 90),   (30, 110),  (1, 1.5)),      # code
    6:  ((2500, 6500),  (75, 20),       (25, 50),   (70, 150),  (2, 3.5)),      # clefG
    7:  ((1500, 5000),  (0, 4),         (25, 50),   (40, 100),  (1.3, 2.1)),    # clefCAlto
    8:  ((1500, 5000),  (0, 4),         (25, 50),   (40, 100),  (1.3, 2.1)),    # clefCTenor
    9:  ((1000, 5000),  (65, 15),       (25, 65),   (30, 80),   (1, 1.5)),      # clefF
    10: (),                                                                     # clefUnpitchedPercussion (no samples)
    11: ((100, 200),    angle_thr_def,  (8, 13),    (10, 16),   (1.25, 1.5)),   # clef8
    12: ((350, 450),    angle_thr_def,  (12, 20),   (20, 30),   (1.4, 1.6)),    # clef15
    13: ((500, 900),    angle_thr_def,  (20, 30),   (25, 40),   ratio_thr_def), # timeSig0
    14: ((350, 700),    (85, 15),       (10, 20),   (25, 40),   (1.5, 2.7)),    # timeSig1
    15: ((550, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1.1, 2)),      # timeSig2
    16: ((450, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1.3, 2)),      # timeSig3
    17: ((500, 900),    (85, 45),       (20, 30),   (25, 40),   (1, 1.8)),      # timeSig4
    18: ((450, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1.2, 1.9)),    # timeSig5
    19: ((500, 800),    (85, 10),       (18, 25),   (25, 40),   (1.2, 2)),      # timeSig6
    20: ((500, 800),    (85, 18),       (18, 25),   (25, 40),   (1.2, 2.2)),    # timeSig7
    21: ((500, 900),    (85, 10),       (18, 25),   (25, 40),   (1.1, 1.9)),    # timeSig8
    22: ((500, 800),    (85, 25),       (18, 25),   (25, 40),   (1.2, 1.9)),    # timeSig9
    23: ((700, 3800),   (0, 25),        (25, 45),   (25, 85),   ratio_thr_def), # timeSigCommon
    24: ((1000, 3800),  (85, 50),       (25, 45),   (30, 90),   ratio_thr_def), # timeSigCutCommon
    25: ((85, 450),     None,           (7, 19),    (10, 25),   (1, 1.8)),      # noteheadBlackOnLine
    26: (),                                                                     # noteheadBlackOnLineSmall (no samples)
    27: ((85, 450),     None,           (7, 19),    (10, 25),   (1, 1.8)),      # noteheadBlackInSpace
    28: (),                                                                     # noteheadBlackInSpaceSmall (no samples)
    29: ((250, 450),    None,           (13, 19),   (16, 27),   ratio_thr_def), # noteheadHalfOnLine
    30: (),                                                                     # noteheadHalfOnLineSmall (no samples)
    31: ((250, 450),    None,           (13, 19),   (16, 27),   ratio_thr_def), # noteheadHalfInSpace
    32: (),                                                                     # noteheadHalfInSpaceSmall (no samples)
    33: ((400, 600),    None,           (13, 19),   (25, 35),   (1.5, 2.2)),    # noteheadWholeOnLine
    34: (),                                                                     # noteheadWholeOnLineSmall (no samples)
    35: ((400, 600),    None,           (13, 19),   (25, 35),   (1.5, 2.2)),    # noteheadWholeInSpace
    36: (),                                                                     # noteheadWholeInSpaceSmall (no samples)
    37: ((650, 1000),   None,           (19, 35),   (30, 50),   (1.1, 2.2)),    # noteheadDoubleWholeOnLine
    38: (),                                                                     # noteheadDoubleWholeOnLineSmall (no samples)
    39: ((650, 1000),   None,           (19, 35),   (30, 50),   (1.1, 2.3)),    # noteheadDoubleWholeInSpace
    40: (),                                                                     # noteheadDoubleWholeInSpaceSmall (no samples)
    41: ((20, 90),      None,           (5, 10),    (5, 12),    (1, 1.6)),      # augmentationDot
    42: ((20, 850),     (89, 1),        (1, 3),     (20, 400),  None),          # stem
    43: ((135, 200),    (65, 80),       (6, 9),     (23, 28),   (3.4, 4.3)),    # tremolo1
    44: ((280, 550),    (60, 10),       (14, 20),   (20, 34),   (1.3, 1.9)),    # tremolo2
    45: ((750, 900),    (0, 0),         (20, 25),   (35, 40),   (1.6, 1.8)),    # tremolo3
    46: ((1100, 1200),  (0, 0),         (20, 25),   (50, 55),   (2.2, 2.4)),    # tremolo4
    47: (),                                                                     # tremolo5 (no samples)
    48: ((50, 900),     (0, 60),        (4, 28),    (10, 60),   (1, 7)),        # flag8thUp
    49: (),                                                                     # flag8thUpSmall (no samples)
    50: ((180, 1100),   (0, 40),        (7, 20),    (22, 65),   (2.5, 5.5)),    # flag16thUp
    51: ((330, 1400),   (85, 10),       (7, 27),    (35, 80),   (1.8, 6)),      # flag32thUp
    52: ((800, 1600),   angle_thr_def,  (11, 20),   (65, 95),   (4.6, 7)),      # flag64thUp
    53: ((950, 1900),   angle_thr_def,  (11, 20),   (80, 120),  (5, 9)),        # flag128thUp
    54: ((50, 900),     None,           (4, 28),    (10, 60),   (1, 7)),        # flag8thDown
    55: (),                                                                     # flag8thDownSmall (no samples)
    56: ((400, 1200),   (85, 45),       (12, 20),   (33, 65),   (2.5, 3.5)),    # flag16thDown
    57: ((600, 1600),   angle_thr_def,  (7, 27),    (35, 80),   (2.8, 4.8)),    # flag32thDown
    58: ((800, 1700),   angle_thr_def,  (11, 22),   (65, 85),   (3.5, 6)),      # flag64thDown
    59: ((950, 2600),   angle_thr_def,  (11, 25),   (80, 120),  (4.1, 7.5)),    # flag128thDown
    60: ((165, 950),    angle_thr_def,  (6, 16),    (20, 65),   (2.3, 4.5)),    # accidentalFlat
    61: (),                                                                     # accidentalFlatSmall (no samples)
    62: ((150, 950),    angle_thr_def,  (6, 13),    (15, 85),   (3.5, 6.8)),    # accidentalNatural
    63: (),                                                                     # accidentalNaturalSmall (no samples)
    64: ((180, 1350),   (85, 10),       (8, 20),    (22, 85),   (2.5, 4.4)),    # accidentalSharp
    65: (),                                                                     # accidentalSharpSmall (no samples)
    66: ((200, 650),    angle_thr_def,  (13, 25),   (14, 32),   (1, 1.5)),      # accidentalDoubleSharp
    67: ((800, 1400),   angle_thr_def,  (18, 40),   (30, 65),   (1, 3)),        # accidentalDoubleFlat
    68: ((400, 950),    angle_thr_def,  (11, 17),   (35, 57),   (2, 5)),        # keyFlat
    69: ((400, 950),    angle_thr_def,  (9, 14),    (40, 80),   (3.5, 7)),      # keyNatural
    70: ((550, 1500),   (85, 10),       (13, 22),   (40, 80),   (2.5, 4.3)),    # keySharp
    71: ((300, 650),    (60, 5),        (10, 25),   (23, 40),   (1.5, 3)),      # articAccentAbove
    72: ((300, 650),    (60, 5),        (10, 25),   (23, 40),   (1.5, 3)),      # articAccentBelow
    73: ((20, 100),     (85, 65),       (5, 12),    (5, 14),    (1, 1.7)),      # articStaccatoAbove
    74: ((20, 100),     (85, 65),       (5, 12),    (5, 14),    (1, 1.7)),      # articStaccatoBelow
    75: ((15, 200),     angle_thr_def,  (1, 8),     (15, 28),   (3, 22)),       # articTenutoAbove
    76: ((15, 200),     angle_thr_def,  (1, 8),     (15, 28),   (3, 22)),       # articTenutoBelow
    77: ((44, 150),     angle_thr_def,  (5, 10),    (9, 20),    (1.5, 3.5)),    # articStaccatissimoAbove
    78: ((44, 150),     angle_thr_def,  (5, 10),    (9, 20),    (1.5, 3.5)),    # articStaccatissimoBelow
    79: ((140, 500),    (85, 30),       (10, 20),   (13, 30),   ratio_thr_def), # articMarcatoAbove
    80: ((140, 500),    (85, 30),       (10, 20),   (13, 30),   ratio_thr_def), # articMarcatoBelow
    81: ((320, 1300),   angle_thr_def,  (10, 30),   (28, 55),   (1.5, 2.2)),    # fermataAbove
    82: ((320, 1300),   angle_thr_def,  (10, 30),   (28, 55),   (1.5, 2.2)),    # fermataBelow
    83: ((50, 300),     (85, 20),       (4, 13),    (13, 28),   (1.5, 4)),      # caesura
    84: ((100, 350),    angle_thr_def,  (7, 18),    (13, 25),   (1, 2.5)),      # restDoubleWhole
    85: ((100, 400),    angle_thr_def,  (7, 16),    (13, 35),   (1, 4.5)),      # restWhole
    86: ((140, 500),    angle_thr_def,  (7, 12),    (20, 45),   (2.2, 4.6)),    # restHalf
    87: ((550, 1200),   (80, 10),       (12, 22),   (40, 60),   (2.2, 4.4)),    # restQuarter
    88: ((350, 700),    (85, 30),       (13, 20),   (24, 35),   (1.4, 2.4)),    # rest8th
    89: ((570, 900),    (1, 25),        (13, 20),   (39, 50),   (2, 3.3)),      # rest16th
    90: ((750, 1300),   (1, 20),        (13, 20),   (52, 70),   (2.5, 4.2)),    # rest32nd
    91: ((950, 1600),   (1, 20),        (13, 22),   (68, 85),   (3.3, 5.4)),    # rest64th
    92: ((1200, 1900),  (1, 20),        (13, 22),   (80, 105),  (4.1, 6.5)),    # rest124th
    93: (),                                                                     # restHNr (no samples)
    94: ((310, 1000),   (0, 22),        (10, 30),   (21, 40),   (1, 3.5)),      # dynamicP
    95: ((300, 800),    angle_thr_def,  (15, 22),   (21, 37),   (1.2, 1.9)),    # dynamicM
    96: ((180, 950),    (85, 45),       (6, 20),    (28, 55),   (2.2, 4.5)),    # dynamicF
    97: ((135, 380),    (85, 25),       (9, 15),    (15, 37),   (1.3, 3.8)),    # dynamicS
    98: ((160, 580),    (85, 30),       (11, 22),   (14, 28),   (1, 1.8)),      # dynamicZ
    99: ((170, 410),    (85, 55),       (9, 18),    (17, 26),   (1, 2.1)),      # dynamicR
    100: (),                                                                    # graceNoteAcciaccaturaStemUp (no samples)
    101: (),                                                                    # graceNoteAppoggiaturaStemUp (no samples)
    102: (),                                                                    # graceNoteAcciaccaturaStemDown (no samples)
    103: (),                                                                    # graceNoteAppoggiaturaStemDown (no samples)
    104: ((620, 1200),  (20, 70),       (20, 34),   (25, 38),   (1, 1.6)),      # ornamentTrill
    105: ((400, 800),   angle_thr_def,  (13, 19),   (30, 45),   (2, 2.5)),      # ornamentTurn
    106: ((420, 1150),  (70, 5),        (14, 27),   (31, 46),   (1.5, 2.5)),    # ornamentTurnInverted
    107: ((330, 850),   (45, 85),       (13, 26),   (24, 38),   ratio_thr_def), # ornamentMordent
    108: ((415, 800),   angle_thr_def,  (19, 25),   (22, 32),   (1, 1.4)),      # stringsDownBow
    109: ((330, 1000),  (85, 20),       (11, 25),   (25, 42),   (1.4, 2.9)),    # stringsUpBow
    110: ((85, 2400),   (85, 25),       (5, 26),    (15, 180),  (1, 25)),       # arpeggiato
    111: ((1550, 3300), angle_thr_def,  (30, 45),   (52, 75),   (1.5, 1.9)),    # keyboardPedalPed
    112: ((520, 4800),  (60, 30),       (22, 50),   (23, 105),  (1, 2.9)),      # keyboardPedalUp
    113: ((260, 300),   angle_thr_def,  (13, 15),   (20, 22),   (1.4, 1.7)),    # tuplet3
    114: ((290, 300),   angle_thr_def,  14,         21,         1.5),           # tuplet6
    115: ((190, 235),   angle_thr_def,  (11, 15),   (15, 20),   (1.1, 1)),      # fingering0
    116: ((100, 230),   (0, 17),        (6, 12),    (13, 22),   (1.6, 3)),      # fingering1
    117: ((145, 245),   (90, 10),       (10, 14),   (14, 21),   (1.1, 2.1)),    # fingering2
    118: ((90, 240),    (90, 10),       (7, 13),    (11, 22),   (1.4, 2.1)),    # fingering3
    119: ((105, 280),   (90, 40),       (9, 16),    (12, 23),   (1.1, 2)),      # fingering4
    120: ((120, 250),   (90, 5),        (9, 14),    (14, 22),   (1.2, 2)),      # fingering5
    121: ((60, 250000), (40, 5),        (4, 100),   (13, 2500), (2, 90)),       # slur
    122: ((50, 11000),  (50, 5),        (4, 9),     (12, 2000), (1.5, 500)),    # beam
    123: ((8, 25000),   (80, 5),        (2, 15),    (4, 2000),  (1.5, 150)),    # tie
    124: ((8000, 10000),(90, 0),        (32, 34),   (260, 280), (8, 9)),        # restHBar
    125: ((240, 70000), angle_thr_def,  (11, 27),   (22, 3500), (1, 200)),      # dynamicCrescendoHairpin
    126: ((240, 70000), angle_thr_def,  (11, 27),   (22, 3500), (1, 200)),      # dynamicDiminuendoHairpin
    127: ((250, 300),   angle_thr_def,  (10, 14),   (12, 24),   (1.3, 2.2)),    # tuplet1
    128: ((250, 330),   angle_thr_def,  (13, 18),   (18, 24),   (1.3, 1.8)),    # tuplet2 (no samples)
    129: ((300, 380),   angle_thr_def,  (13, 20),   (15, 24),   (1, 1.5)),      # tuplet4
    130: ((250, 330),   angle_thr_def,  (13, 18),   (18, 24),   (1.1, 1.5)),    # tuplet5
    131: ((250, 330),   angle_thr_def,  (13, 18),   (18, 24),   (1.3, 1.8)),    # tuplet7
    132: ((250, 330),   angle_thr_def,  (13, 18),   (18, 24),   (1.3, 1.8)),    # tuplet8
    133: ((280, 350),   angle_thr_def,  (13, 18),   (18, 24),   (1.3, 1.8)),    # tuplet9
    134: ((580, 10000), (75, 15),       (10, 30),   (55, 700),  (4, 80)),       # tupletBracket
    135: ((271, 250000),angle_thr_def,  (1, 70),    (400, 4500),(6, 4500)),     # staff
    136: ((50, 25000),  (80, 10),       (1, 17),    (10, 2500), (2, 2500)),     # ottavaBracket
    # 137-208 are muscima++
}

def get_thresholds(cat_stats: dict) -> Dict[str, Tuple[float, float]]:
    def expand_threshold(thresh) -> Tuple[float, float]:
        if thresh is None:
            return None, None
        if not isinstance(thresh, tuple):
            thresh = thresh, thresh
        low_thr, high_thr = thresh
        return low_thr, high_thr
    thr_list = thresholds.get(cat_stats['id'], [])
    out = defaultdict(lambda: (None, None))
    for thr_cls, threshold in zip(threshold_classes, thr_list):
        klass_stat = cat_stats[thr_cls]
        out[thr_cls] = expand_threshold(threshold, klass_stat['median'], klass_stat['std'])
    return {cls:out[cls] for cls in threshold_classes}
