import json
from argparse import ArgumentParser
from collections import defaultdict

from PIL import Image, ImageDraw
from PIL import ImageFont
from typing import Tuple, List, Dict, Optional

from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure

from mmdet.datasets import DeepScoresV2Dataset
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt, axes

from mmdet.datasets.pipelines.transforms import OBBox
from obb_anns import OBBAnns

parser = ArgumentParser(description='Deepscores statistics')
parser.add_argument('-c', '--compile', dest='compile', action='store_true', default=False,
                    help='Compiles statistics')
parser.add_argument('-p', '--plot-stats', dest='plot_stats', action='store_true', default=False,
                    help='Plots statistics')
parser.add_argument('-f', '--flag-outliers', dest='flag_outliers', action='store_true', default=False,
                    help='Flags outliers using past statistics')
parser.add_argument('-g', '--geogebra', dest='to_geogebra', action='store', default=None,
                    help='Converts json annotation to geogebra inputs')
parser.add_argument('-a', '--fix-annotations', dest='fix_annotations', action='store_true', default=False,
                    help='Fixes the annotations which have an area of 0')
args = parser.parse_args()

cat_selection = {'94'}
threshold_classes = ['area', 'angle', 'l1', 'l2', 'edge-ratio']
crit_selection = {*threshold_classes}
crit_selection = {}
area_thr_def = (1, 1.0)
angle_thr_def = (85, 5)  # Between 5 and 85°
l1_thr_def = (1, 1.0)
l2_thr_def = (1, 1.0)
ratio_thr_def = (1, 2)
thresholds = {
    # tuple: Upper and lower bound. vs single value: symmetric bounds
    # float: std deviation factor
    # int: absolute threshold
    # If high value first: Inverted threshold (mainly used for angles)
    #   Area,           Angle,          L1,         L2,         Ratio
    1:  ((1000, 30000), angle_thr_def,  (5, 50),    (10, 600),  (5, 50)),       # brace
    2:  ((23, 258),     0,              (1, 5),     (17, 100),  (5, 50)),       # ledgerLine
    3:  ((20, 90),      None,           (4, 10),    (4, 13),    ratio_thr_def), # repeatDot
    4:  ((1500, 9000),  angle_thr_def,  (25, 90),   (30, 110),  ratio_thr_def), # segno
    5:  ((1000, 9000),  None,           (25, 90),   (30, 110),  ratio_thr_def), # code
    6:  ((2500, 6500),  (75, 20),       (25, 50),   (70, 150),  (2, 4)),        # clefG
    7:  ((1500, 5000),  (0, 4),         (25, 50),   (40, 100),  ratio_thr_def), # clefCAlto
    8:  ((1500, 5000),  (0, 4),         (25, 50),   (40, 100),  ratio_thr_def), # clefCTenor
    9:  ((1000, 5000),  (80, 25),       (25, 65),   (30, 80),   ratio_thr_def), # clefF
    # 10 has been deleted
    11: ((100, 200),    angle_thr_def,  (8, 13),    (10, 16),   ratio_thr_def), # clef8
    12: ((350, 450),    angle_thr_def,  (12, 20),   (20, 30),   ratio_thr_def), # clef15
    13: ((500, 900),    angle_thr_def,  (20, 30),   (25, 40),   ratio_thr_def), # timeSig0
    14: ((350, 700),    angle_thr_def,  (10, 20),   (25, 40),   (1, 3)),        # timeSig1
    15: ((550, 800),    angle_thr_def,  (18, 25),   (25, 40),   ratio_thr_def), # timeSig2
    16: ((450, 800),    angle_thr_def,  (18, 25),   (25, 40),   ratio_thr_def), # timeSig3
    17: ((500, 900),    angle_thr_def,  (20, 30),   (25, 40),   ratio_thr_def), # timeSig4
    18: ((450, 800),    angle_thr_def,  (18, 25),   (25, 40),   ratio_thr_def), # timeSig5
    19: ((500, 800),    angle_thr_def,  (18, 25),   (25, 40),   ratio_thr_def), # timeSig6
    20: ((500, 800),    (70, 5),        (18, 25),   (25, 40),   (1, 3)),        # timeSig7
    21: ((500, 900),    angle_thr_def,  (18, 25),   (25, 40),   ratio_thr_def), # timeSig8
    22: ((500, 800),    angle_thr_def,  (18, 25),   (25, 40),   ratio_thr_def), # timeSig9
    23: ((700, 3800),   (70, 10),       (25, 45),   (25, 85),   ratio_thr_def), # timeSigCommon
    24: ((1000, 3800),  None,           (25, 45),   (30, 90),   ratio_thr_def), # timeSigCutCommon
    25: ((85, 450),     None,           (7, 19),    (10, 25),   ratio_thr_def), # noteheadBlackOnLine
    # 26 has been deleted
    27: ((85, 450),     None,           (7, 19),    (10, 25),   ratio_thr_def), # noteheadBlackInSpace
    # 28 has been deleted
    29: ((250, 450),    None,           (13, 19),   (16, 27),   ratio_thr_def), # noteheadHalfOnLine
    # 30 has been deleted
    31: ((250, 450),    None,           (13, 19),   (16, 27),   ratio_thr_def), # noteheadHalfInSpace
    # 32 has been deleted
    33: ((400, 600),    None,           (13, 19),   (25, 35),   (1, 2.2)),      # noteheadWholeOnLine
    # 34 has been deleted
    35: ((400, 600),    None,           (13, 19),   (25, 35),   (1, 2.2)),      # noteheadWholeInSpace
    # 36 has been deleted
    37: ((650, 1000),   angle_thr_def,  (19, 35),   (30, 50),   ratio_thr_def), # noteheadDoubleWholeOnLine
    # 38 has been deleted
    39: ((650, 1000),   angle_thr_def,  (19, 35),   (30, 50),   ratio_thr_def), # noteheadDoubleWholeInSpace
    # 40 has been deleted
    41: ((20, 90),      None,           (5, 10),    (5, 12),    ratio_thr_def), # augmentationDot
    42: ((20, 850),     (89, 1),        (1, 3),     (20, 400),  None),          # stem
    # 43 has been deleted
    44: ((280, 550),    (0, 25),        (14, 20),   (20, 34),   ratio_thr_def), # tremolo2
    45: ((750, 900),    (0, 0),         (20, 25),   (35, 40),   ratio_thr_def), # tremolo3
    46: ((1100, 1200),  (0, 0),         (20, 25),   (50, 55),   (2, 3)),        # tremolo4
    # 47 has been deleted
    48: ((50, 900),     None,           (4, 28),    (10, 60),   (1, 7)),        # flag8thUp
    # 49 has been deleted
    50: ((180, 1100),   (0, 40),        (7, 20),    (22, 65),   (2, 6)),        # flag16thUp
    51: ((330, 1400),   (75, 5),        (7, 27),    (35, 80),   (1, 6)),        # flag32thUp
    52: ((800, 1600),   angle_thr_def,  (11, 20),   (65, 95),   (4, 7)),        # flag64thUp
    53: ((950, 1900),   angle_thr_def,  (11, 20),   (80, 120),  (4, 9)),        # flag128thUp
    54: ((50, 900),     None,           (4, 28),    (10, 60),   (1, 7)),        # flag8thDown
    # 55 has been deleted
    56: ((400, 1200),   (45, 5),        (12, 20),   (33, 65),   (2, 4)),        # flag16thDown
    57: ((600, 1600),   (75, 5),        (7, 27),    (35, 80),   (1, 6)),        # flag32thDown
    58: ((800, 1700),   angle_thr_def,  (11, 22),   (65, 85),   (3, 6)),        # flag64thDown
    59: ((950, 2600),   angle_thr_def,  (11, 25),   (80, 120),  (4, 8)),        # flag128thDown
    60: ((165, 950),    angle_thr_def,  (6, 16),    (20, 65),   (2, 5)),        # accidentalFlat
    # 61 has been deleted
    62: ((170, 950),    (80, 5),        (6, 13),    (15, 85),   (3, 7)),        # accidentalNatural
    # 63 has been deleted
    64: ((180, 1350),   (80, 5),        (10, 20),   (22, 85),   (2, 5)),        # accidentalSharp
    # 65 has been deleted
    66: ((200, 650),    (80, 5),        (13, 25),   (14, 32),   ratio_thr_def), # accidentalDoubleSharp
    67: ((800, 1400),   angle_thr_def,  (18, 40),   (30, 65),   (1, 3)),        # accidentalDoubleFlat
    68: ((400, 950),    angle_thr_def,  (11, 17),   (35, 57),   (2, 5)),        # keyFlat
    69: ((400, 950),    angle_thr_def,  (9, 14),    (40, 80),   (3, 7)),        # keyNatural
    70: ((550, 1500),   angle_thr_def,  (13, 22),   (40, 80),   (2, 5)),        # keySharp
    71: ((300, 650),    (70, 30),       (10, 25),   (23, 40),   (1, 3)),        # articAccentAbove
    72: ((300, 650),    (70, 30),       (10, 25),   (23, 40),   (1, 3)),        # articAccentBelow
    73: ((20, 100),     None,           (5, 10),    (4, 11),    (1, 3.0)),      # articStaccatoAbove
    74: ((20, 100),     None,           (5, 10),    (4, 11),    (1, 3.0)),      # articStaccatoBelow
    75: ((15, 200),     (85, 10),       (1, 8),     (15, 28),   (3, 22)),       # articTenutoAbove
    76: ((15, 200),     (85, 10),       (1, 8),     (15, 28),   (3, 22)),       # articTenutoBelow
    77: ((44, 150),     angle_thr_def,  (5, 10),    (9, 20),    (1, 4)),        # articStaccatissimoAbove
    78: ((44, 150),     angle_thr_def,  (5, 10),    (9, 20),    (1, 4)),        # articStaccatissimoBelow
    79: ((140, 500),    angle_thr_def,  (10, 20),   (13, 30),   ratio_thr_def), # articMarcatoAbove
    80: ((140, 500),    angle_thr_def,  (10, 20),   (13, 30),   ratio_thr_def), # articMarcatoBelow
    81: ((320, 1300),   angle_thr_def,  (10, 30),   (28, 55),   (1.6, 2)),      # fermataAbove
    82: ((320, 1300),   angle_thr_def,  (10, 30),   (28, 55),   (1.6, 2)),      # fermataBelow
    83: ((50, 300),     (70, 20),       (4, 13),    (13, 28),   (1.6, 4)),      # caesura
    84: ((100, 350),    angle_thr_def,  (7, 18),    (13, 25),   (1, 3)),        # restDoubleWhole
    85: ((100, 400),    angle_thr_def,  (7, 16),    (13, 35),   (1, 4)),        # restWhole
    86: ((140, 250),    angle_thr_def,  (7, 12),    (20, 30),   (2, 4)),        # restHalf
    87: ((550, 1200),   (80, 10),       (12, 22),   (40, 60),   (2, 4)),        # restQuarter
    88: ((350, 700),    (60, 5),        (13, 20),   (24, 35),   (1.4, 2.4)),    # rest8th
    89: ((570, 900),    (65, 90),       (13, 20),   (39, 50),   (2, 3.3)),      # rest16th
    90: ((750, 1300),   (65, 90),       (13, 20),   (52, 70),   (2.5, 4.2)),    # rest32nd
    91: ((950, 1600),   (65, 90),       (13, 22),   (68, 85),   (3.3, 5.4)),    # rest64th
    92: ((1200, 1900),  (65, 90),       (13, 22),   (80, 105),  (4.1, 6.5)),    # rest124th
    # 93 has been deleted
    94: ((310, 1000),   (65, 5),        (10, 30),   (21, 40),   (1, 3.4)),      # dynamicP
    95: (),  # dynamicM
    96: (),  # dynamicF
    97: (),  # dynamicS
    98: (),  # dynamicZ
    99: (),  # dynamicR
    # 100 has been deleted
    # 101 has been deleted
    # 102 has been deleted
    # 103 has been deleted
    104: (),  # ornamentTrill
    105: (),  # ornamentTurn
    106: (),  # ornamentTurnInverted
    107: (),  # ornamentMordent
    108: (),  # stringsDownBow
    109: (),  # stringsDownBow
    110: (),  # arpeggiato
    111: (),  # keaboardPedalPed
    112: (),  # keyboardPedalUp
    113: (),  # tuplet3
    114: (),  # tuplet6
    115: (),  # fingering0
    116: (),  # fingering1
    117: (),  # fingering2
    118: (),  # fingering3
    119: (),  # fingering4
    120: (),  # fingering5
    121: (),  # slur
    # 122 has been deleted
    123: (),  # tie
    124: (),  # restHBar
    125: (),  # dynamicCrescendoHairpin
    126: (),  # dynamicDiminuendoHairpin
    # 127 has been deleted
    129: (),  # tuplet4
    # 130 has been deleted
    # 131 has been deleted
    # 132 has been deleted
    # 133 has been deleted
    134: (),  # tupletBracket
    # 135 has been deleted
    136: (),  # ottavaBracket
}
def get_thresholds(cat_stats: dict) -> Dict[str, Tuple[float, float]]:
    def expand_threshold(threshold, median: float, std: float) -> Tuple[float, float]:
        if threshold is None:
            return None, None
        if not isinstance(threshold, tuple):
            threshold = threshold, threshold
        low_thr, high_thr = threshold
        # if isinstance(low_thr, float):
        #     low_thr *= std
        #     low_thr = median - low_thr
        # if isinstance(high_thr, float):
        #     high_thr *= std
        #     high_thr = median + high_thr
        return low_thr, high_thr
    thr_list = thresholds.get(cat_stats['id'], [])
    out = defaultdict(lambda: (None, None))
    for thr_cls, threshold in zip(threshold_classes, thr_list):
        klass_stat = cat_stats[thr_cls]
        # median = klass_stat['median'] if thr_cls != 'angle' else 0.0
        # std = klass_stat['std'] if thr_cls != 'angle' else 1.0
        # out[thr_cls] = expand_threshold(threshold, median, std)
        out[thr_cls] = expand_threshold(threshold, klass_stat['median'], klass_stat['std'])
    return {cls:out[cls] for cls in threshold_classes}


def is_attribute_an_outlier(cat_stats: dict, cat_thresholds: Dict[str, Tuple[float, float]], cls: str, value: float) -> bool:
    if len(crit_selection) != 0 and cls not in crit_selection:
        return False
    low_thr, high_thr = cat_thresholds.get(cls, (None, None))
    if low_thr is None or high_thr is None:
        return False
    if low_thr > high_thr:  # Inverted threshold
        return high_thr < value < low_thr
    return not (low_thr <= value <= high_thr)

def flag_outlier(obbox: np.ndarray, cat_id: int, stats: dict) -> Dict[str, float]:
    if isinstance(obbox, list):
        obbox = np.array(obbox)
    obbox = np.array(obbox).reshape((4, 2))
    cat_stats = stats[str(cat_id)]
    cat_thrs = get_thresholds(cat_stats)
    # check all attributes
    reasons = {}
    area = OBBox.get_area(obbox)
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'area', area):
        reasons['area'] = area
    angle = (OBBox.get_angle(obbox)) % 90
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'angle', angle):
        reasons['angle'] = angle
    l1 = np.linalg.norm(obbox[0] - obbox[1])
    l2 = np.linalg.norm(obbox[1] - obbox[2])
    l1, l2 = min(l1, l2), max(l1, l2)
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'l1', l1):
        reasons['l1'] = l1
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'l2', l2):
        reasons['l2'] = l2
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'edge-ratio', l2 / l1):
        reasons['edge-ratio'] = l2 / l1
    return reasons

def plot_attribute(ax: Axes, cat_stats: dict, cat_thrs: dict, thr_cls: str):
    cls_stats = cat_stats[thr_cls]
    values = cls_stats['sorted']
    n = len(values)
    n_outliers = sum(map(lambda value: is_attribute_an_outlier(cat_stats, cat_thrs, thr_cls, value), values))
    median = cls_stats['median']
    mean = cls_stats['mean']
    std = cls_stats['std']
    minimum = round(cls_stats['min'], 2)
    maximum = round(cls_stats['max'], 2)
    low_thr, high_thr = cat_thrs[thr_cls]
    low_thr = round(low_thr, 2) if low_thr is not None else None
    high_thr = round(high_thr, 2) if high_thr is not None else None
    if low_thr is not None and high_thr is not None:
        if low_thr > high_thr:
            bounds = f"\n]{high_thr}, {low_thr}["
        else:
            bounds = f"\n[{low_thr}, {high_thr}]"
    else:
        bounds = ''
    ax.set_title(f"{thr_cls}: {n_outliers} outliers [{minimum}, {maximum}]{bounds}")
    ax.set_xlabel(f'Index of sorted {thr_cls}')
    ax.set_ylabel(f'{thr_cls}, sorted')
    ax.set_ylim(ymin=min(values), ymax=max(values))
    ax.grid(True)
    ax.plot(range(n)[::-1], values)
    ax.plot([0, n], [median, median], color='#20dd50')
    ax.plot([0, n], [mean, mean], color='#55ffaa')
    ax.plot([0, n], [median + std, median + std], color='#ff9050')  # Upper std
    ax.plot([0, n], [median - std, median - std], color='#ff9050')  # Lower std
    ax.plot([0, n], [high_thr, high_thr], color='#dd4020')  # Upper bound
    ax.plot([0, n], [low_thr, low_thr], color='#dd4020')  # Lower bound

def plot(cat_id: int, cat_stats: dict):
    fig: Figure
    axs: List[Axes]
    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    cat_thrs = get_thresholds(cat_stats)
    for ax, thr_cls in zip(axs.reshape((6,)), threshold_classes):
        plot_attribute(ax, cat_stats, cat_thrs, thr_cls)
    fig.suptitle(f"[{cat_id}] {cat_stats['name']}")
    fig.show()

STAT_FILE = 'stats.json'
if args.plot_stats:
    print("Loading stats")
    if 'stats' not in globals():
        with open(STAT_FILE, 'r') as fp:
            stats = json.load(fp)
    for cat_id, sts in sorted(stats.items(), key=lambda kvp: int(kvp[0])):
        if len(cat_selection) > 0 and cat_id in cat_selection:
            plot(cat_id, sts)

if args.to_geogebra:
    print("Converting to geogebra")
    print("\033[1m")
    data = json.loads(args.to_geogebra)
    a_bbox = np.array(data['a_bbox']).reshape((2, 2))
    o_bbox = np.array(data['o_bbox']).reshape((4, 2))
    i = 65
    chars = []
    for p in a_bbox:
        char = chr(i)
        # print(f'ggbApplet.deleteObject("{char}")')
        print(f'\033[31mggbApplet.evalCommand("{char}=({p[0]}, {-p[1]})")\033[39m')
        chars.append(char)
        i += 2
    print('ggbApplet.evalCommand("a : Line(A, xAxis)")')
    print('ggbApplet.evalCommand("b : Line(C, xAxis)")')
    print('ggbApplet.evalCommand("c : Line(A, yAxis)")')
    print('ggbApplet.evalCommand("d : Line(C, yAxis)")')
    print('ggbApplet.evalCommand("B = Intersect(a, d)")')
    print('ggbApplet.evalCommand("D = Intersect(b, c)")')
    print(f'ggbApplet.evalCommand("abbox = Polygon(A, B, C, D)")')
    chars = []
    for p in o_bbox:
        char = chr(i)
        # print(f'ggbApplet.deleteObject("{char}")')
        print(f'\033[31mggbApplet.evalCommand("{char}=({p[0]}, {-p[1]})")\033[39m')
        chars.append(char)
        i += 1
    print(f'ggbApplet.evalCommand("obbox = Polygon({", ".join(chars)})")')
    print("\033[m")

if not args.flag_outliers and not args.compile and not args.fix_annotations:
    exit(0)

pipeline = [
    {'type': 'LoadImageFromFile'},
    {'type': 'LoadAnnotations', 'with_bbox': True},
    {'type': 'RandomCrop', 'crop_size': (1400, 1400), 'threshold_rel': 0.6, 'threshold_abs': 20.0},
    {'type': 'RotatedResize', 'img_scale': (1024, 1024), 'keep_ratio': True},
    {'type': 'RotatedRandomFlip', 'flip_ratio': 0.0},
    {'type': 'Normalize', 'mean': [240, 240, 240], 'std': [57, 57, 57], 'to_rgb': False},
    {'type': 'Pad', 'size_divisor': 32},
    {'type': 'DefaultFormatBundle'},
    {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}
]
dataset_train = DeepScoresV2Dataset(
    ann_file='deepscores_train.json',
    img_prefix='data/deep_scores_dense/images/',
    pipeline=pipeline,
    use_oriented_bboxes=True,
)
dataset_test = DeepScoresV2Dataset(
    ann_file='deepscores_test.json',
    img_prefix='data/deep_scores_dense/images/',
    pipeline=pipeline,
    use_oriented_bboxes=True,
)
cat_info = dataset_train.obb.cat_info

def gather_stats(dataset: DeepScoresV2Dataset, stats: dict):
    for cat_id, o_bbox in zip(dataset.obb.ann_info['cat_id'], dataset.obb.ann_info['o_bbox']):
        cat = str(cat_id[0])
        obbox = np.array(o_bbox).reshape((4, 2))
        attributes = stats.get(cat, defaultdict(lambda: []))
        attributes['area'].append(OBBox.get_area(obbox))
        attributes['angle'].append((OBBox.get_angle(obbox)) % 90)
        l1 = np.linalg.norm(obbox[0] - obbox[1])
        l2 = np.linalg.norm(obbox[1] - obbox[2])
        l1, l2 = min(l1, l2), max(l1, l2)
        attributes['l1'].append(l1)
        attributes['l2'].append(l2)
        attributes['edge-ratio'].append(l2 / l1)
        stats[cat] = attributes

def compile_stats(stats: dict, cat_info: dict):
    def to_dict(values: List[float]) -> dict:
        return {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'length': len(values),
            'sorted': sorted(values, reverse=True)
        }
    for cat, data in stats.items():
        stats[cat] = {
            'id': int(cat),
            'name': cat_info[int(cat)]['name'],
            'area': to_dict(data['area']),
            'angle': to_dict(data['angle']),
            'l1': to_dict(data['l1']),
            'l2': to_dict(data['l2']),
            'edge-ratio': to_dict(data['edge-ratio']),
        }

def filter_bboxes(dataset: DeepScoresV2Dataset, stats: dict) -> dict:
    imgs = {}
    img_dir = osp.join(osp.split(dataset.obb.ann_file)[0], 'images')
    for cat_id, a_bbox, o_bbox, img_id, idx in zip(
            dataset.obb.ann_info['cat_id'],
            dataset.obb.ann_info['a_bbox'],
            dataset.obb.ann_info['o_bbox'],
            dataset.obb.ann_info['img_id'],
            dataset.obb.ann_info.index,
    ):
        cat = int(cat_id[0])
        if len(cat_selection) > 0 and str(cat) in cat_selection:
            flags = flag_outlier(o_bbox, cat, stats)
            if len(flags) > 0:
                if img_id not in imgs.keys():
                    img_info, _ = dataset.obb.get_img_ann_pair(idxs=None, ids=[int(img_id)])
                    filename = img_info[0]['filename']
                    imgs[img_id] = (osp.join(img_dir, filename), [])
                imgs[img_id][1].append((cat, a_bbox, o_bbox, idx, flags))
    return imgs

def draw_outliers(imgs: dict, cat_info: dict) -> dict:
    outlier_stats = {}
    for img_id, (img_fp, bboxes) in sorted(imgs.items(), reverse=True, key=lambda kvp: len(kvp[1])):
        img = Image.open(img_fp)
        draw = ImageDraw.Draw(img, 'RGBA')
        cats = set(map(lambda tpl: tpl[0], bboxes))
        print(f"Visualized {len(bboxes)} outliers in {osp.basename(img_fp)} from the categories: {cats}")
        for cat, a_bbox, bbox, idx, flags in bboxes:
            draw.rectangle(a_bbox, outline='#223CF088', width=3)
            draw.line(bbox + bbox[:2], fill='#F03C2288', width=3)
            flag_str = '\n'.join(map(lambda kv: f"{kv[0]}: {round(kv[1], 2)}", list(flags.items())))
            text = f"[{idx}] {cat_info[cat]['name']}({cat}):\n{flag_str}"
            print(f"  * {text.replace(chr(10), ' ')}")
            # Get text position
            bbox_np = np.array(bbox).reshape((4, 2))
            position = int(bbox_np.max(axis=0)[0]), int(bbox_np.min(axis=0)[1])
            # Get text dimensions for gray bg box
            x, y = 0.0, 0.0
            for line in text.splitlines():
                x1, y1 = ImageFont.load_default().getsize(line)
                x = max(x, x1)
                y += y1 + 3  # 3: interline spacing
            x += position[0] + 4
            y += position[1] + 4
            # Draw gray bg for text
            draw.rectangle((position[0], position[1], x, y), fill='#DCDCDC88')
            draw.text((position[0] + 2, position[1] + 2), text, '#F03C22')
            outlier_stats[cat] = outlier_stats.get(cat, 0) + 1
        img.save(osp.join('out_debug', osp.basename(img_fp)))
    return outlier_stats


if args.compile:
    stats = defaultdict(lambda: {})
    print("Gathering the stats")
    gather_stats(dataset_train, stats)
    gather_stats(dataset_test, stats)
    print("Calculating the stats")
    compile_stats(stats, dataset_test.obb.cat_info)
    print("Saving the stats")
    with open(STAT_FILE, 'w') as fp:
        json.dump(stats, fp, indent=4)
    print("Done")

if args.flag_outliers:
    if 'stats' not in globals():
        print("Loading stats")
        with open(STAT_FILE, 'r') as fp:
            stats = json.load(fp)
    imgs = {}
    print("Searching for outliers")
    imgs_train = filter_bboxes(dataset_train, stats)
    imgs_test = filter_bboxes(dataset_test, stats)
    print("Drawing outliers and saving them to disk")
    outlier_stats_train = draw_outliers(imgs_train, cat_info)
    print(f"{'#'*10} TEST DATASET {'#'*10}")
    outlier_stats_test = draw_outliers(imgs_test, cat_info)
    print()
    print("[Train stats]")
    total = 0
    for cat, number in sorted(outlier_stats_train.items(), reverse=True, key=lambda kvp: kvp[1]):
        print(f"{cat_info[cat]['name']} ({cat}): {number}")
        total += number
    print(f"Total possible outliers detected: {total}")
    print()
    print("[Test stats]")
    total = 0
    for cat, number in sorted(outlier_stats_test.items(), reverse=True, key=lambda kvp: kvp[1]):
        print(f"{cat_info[cat]['name']} ({cat}): {number}")
        total += number
    print(f"Total possible outliers detected: {total}")

def fix_annotations(anns: OBBAnns):
    def fix_ann(bbox: list) -> list:
        bbox = list(map(int, bbox))
        bbox = np.array([bbox[0::2], bbox[1::2]]).T
        if OBBox.get_area(bbox) < 1.0:
            if np.all(bbox[:,0] == np.full((4,), bbox[0,0])):  # all X's are the same
                bbox[1:3,0] = bbox[1:3,0] + np.ones((2,))
            if np.all(bbox[:,1] == np.full((4,), bbox[0,1])):  # all Ys are the same
                bbox[2:,1] = bbox[2:,1] + np.ones((2,))
            if np.all(bbox[0::2,:] == bbox[1::2,:]):  # weird constellation
                bbox[1, 1] = bbox[2, 1]
                bbox[3, 1] = bbox[0, 1]
            print(".", end='')
        return list(map(int, bbox.reshape((8,))))
    def per_row(x):
        abbox, obbox = x[0], x[1]
        # make abbox into a 8-tuple like obbox
        abbox = [abbox[2], abbox[3], abbox[2], abbox[1], abbox[0], abbox[1], abbox[0], abbox[3]]
        abbox = fix_ann(abbox)
        x[0] = [abbox[0], abbox[1], abbox[4], abbox[5]]
        obbox = fix_ann(obbox)
        x[1] = obbox
        x[3] = int(OBBox.get_area(np.array([obbox[0::2], obbox[1::2]]).T))
        return x
    anns.ann_info = anns.ann_info.apply(per_row, axis=1, raw=True)
    print()

if args.fix_annotations:
    print('[TRAIN] Fixing annotations')
    fix_annotations(dataset_train.obb)
    print('[TRAIN] Saving dataset to deepscores_train.fixed.json')
    dataset_train.obb.save_annotations('deepscores_train.fixed.json')
    print('[TEST] Fixing annotations')
    fix_annotations(dataset_test.obb)
    print('[TEST] Saving dataset to deepscores_test.fixed.json')
    dataset_test.obb.save_annotations('deepscores_test.fixed.json')
    print('Done')
