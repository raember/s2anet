import json
from argparse import ArgumentParser
from collections import defaultdict

from PIL import Image, ImageDraw
from PIL import ImageFont
from typing import Tuple, List, Dict

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

cat_selection = {'19'}
crit_selection = {}
threshold_classes = ['area', 'angle', 'l1', 'l2', 'edge-ratio']
area_thr_def = (1, 1.0)
angle_thr_def = (85, 5)  # Between 5 and 85°
l1_thr_def = (1, 1.0)
l2_thr_def = (1, 1.0)
ratio_thr_def = (1, 1.0)
thresholds = {
    # tuple: Upper and lower bound. vs single value: symmetric bounds
    # float: std deviation factor
    # int: absolute threshold
    # If high value first: Inverted threshold (mainly used for angles)
    #   Area,           Angle,          L1,         L2,         Ratio
    1:  ((1000, 30000), angle_thr_def,  (5, 50),    (10, 600),  (5, 50)),       # brace
    2:  ((23, 258),     0,              (1, 5),     (17, 100),  (5, 50)),       # ledgerLine
    3:  ((20, 90),      None,           (4, 10),    (4, 13),    (1, 2)),        # repeatDot
    4:  ((1500, 9000),  angle_thr_def,  (25, 90),   (30, 110),  ratio_thr_def), # segno
    5:  ((1000, 9000),  None,           (25, 90),   (30, 110),  ratio_thr_def), # code
    6:  ((2500, 6500),  (75, 20),       (25, 50),   (70, 150),  (2, 4)),        # clefG
    7:  ((1500, 5000),  (0, 4),         (25, 50),   (40, 100),  (1, 2)),        # clefCAlto
    8:  ((1500, 5000),  (0, 4),         (25, 50),   (40, 100),  (1, 2)),        # clefCTenor
    9:  ((1000, 5000),  (80, 25),       (25, 65),   (30, 80),   (1, 2)),        # clefF
    # 10 has been deleted
    11: ((100, 200),    angle_thr_def,  (8, 13),    (10, 16),   (1, 2)),        # clef8
    12: ((350, 450),    angle_thr_def,  (12, 20),   (20, 30),   (1, 2)),        # clef15
    13: ((500, 900),    angle_thr_def,  (20, 30),   (25, 40),   (1, 2)),        # timeSig0
    14: ((350, 700),    angle_thr_def,  (10, 20),   (25, 40),   (1, 3)),        # timeSig1
    15: ((550, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1, 2)),        # timeSig2
    16: ((450, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1, 2)),        # timeSig3
    17: ((500, 900),    angle_thr_def,  (20, 30),   (25, 40),   (1, 2)),        # timeSig4
    18: ((450, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1, 2)),        # timeSig5
    19: ((500, 800),    angle_thr_def,  (18, 25),   (25, 40),   (1, 2)),        # timeSig6
    25: ((5.5, 2.2),),  # noteheadBlackOnLine
    27: ((5.5, 1.5),),  # noteheadBlackInSpace
    # 31: ((6.0, 3.0)),  # noteheadHalfInSpace
    # 33: ((8.0)),  # noteheadWholeOnLine
    # 42: ((2.0, 15.0)),  # stem
    64: (3.0,),  # accidentialSharp
    70: ((1.0, 5.0),),  # keySharp
    85: ((1.5, 6.0),),  # restWhole
    88: ((1.8, 2.5),),  # rest8th
    90: (2.0,),  # rest32nd
    #113: (10.0),  # tuplet3
    122: ((1.1, 17.0),),  # beam
    135: ((4.5, 4.5),),  # staff
}
def get_thresholds(cat_stats: dict) -> Dict[str, Tuple[float, float]]:
    def expand_threshold(threshold, median: float, std: float) -> Tuple[float, float]:
        if threshold is None:
            return None, None
        if not isinstance(threshold, tuple):
            threshold = threshold, threshold
        low_thr, high_thr = threshold
        if isinstance(low_thr, float):
            low_thr *= std
            low_thr = median - low_thr
        if isinstance(high_thr, float):
            high_thr *= std
            high_thr = median + high_thr
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


deviation = {
    2: (2.2, 10.0),  # ledgerLine
    25: (5.5, 2.2),  # noteheadBlackOnLine
    27: (5.5, 1.5),  # noteheadBlackInSpace
    # 31: (6.0, 3.0),  # noteheadHalfInSpace
    # 33: 8.0,  # noteheadWholeOnLine
    # 42: (2.0, 15.0),  # stem
    64: 3.0,  # accidentialSharp
    70: (1.0, 5.0),  # keySharp
    85: (1.5, 6.0),  # restWhole
    88: (1.8, 2.5),  # rest8th
    90: 2.0,  # rest32nd
    #113: 10.0,  # tuplet3
    122: (1.1, 17.0),  # beam
    135: (4.5, 4.5),  # staff
}
ignore = {
    1,  # brace
    3,  # repeatDot
    4,  # segno
    5,  # coda
    6,  # clefG
    7,  # clefCAlto
    8,  # clefCTenor
    9,  # clefF
    11,  # clef8
    13,  # timeSig0
    14,  # timeSig1
    15,  # timeSig2
    16,  # timeSig3
    17,  # timeSig4
    18,  # timeSig18
    19,  # timeSig6
    20,  # timeSig7
    21,  # timeSig8
    22,  # timeSig9
    23,  # timeSigCommon
    24,  # timeSigCutCommon
    29,  # noteheadHalfOnLine
    31,  # noteheadHalfInSpace
    33,  # noteheadWholeOnLine
    35,  # noteheadWholeInSpace
    37,  # noteheadDoubleWholeOnLine
    39,  # noteheadDoubleWholeInSpace
    41,  # augmentationDot
    42,  # stem
    43,  # tremolo1
    44,  # tremolo2
    45,  # tremolo3
    46,  # tremolo4
    48,  # flag8thUp
    50,  # flag16thUp
    51,  # flag32ndUp
    52,  # flag64thUp
    53,  # flag128thUp
    54,  # flag8thDown
    56,  # flag16thDown
    57,  # flag32ndDown
    58,  # flag64thDown
    59,  # flag128thDown
    60,  # accidentialFlat
    62,  # accidentialNatural
    66,  # accidentialDoubleSharp
    67,  # accidentialDoubleFlat
    68,  # keyFlat
    69,  # keyNatural
    71,  # articAccentAbove
    72,  # articAccentBelow
    73,  # articStaccatoAbove
    74,  # articStaccatoBelow
    75,  # articTenutoAbove
    76,  # articTenutoBelow
    77,  # articStaccatissimoAbove
    78,  # articStaccatissimoBelow
    79,  # articMarcatoAbove
    80,  # articMarcatoBelow
    81,  # fermataAbove
    82,  # fermataBelow
    83,  # caesura
    84,  # restDoubleWhole
    86,  # restHalf
    87,  # restQuarter
    89,  # rest16th
    91,  # rest64th
    92,  # rest124th
    94,  # dynamicP
    95,  # dynamicM
    96,  # dynamicF
    97,  # dynamicS
    98,  # dynamicZ
    99,  # dynamicR
    104,  # ornamentTrill
    105,  # ornamentTurn
    106,  # ornamentTurnInverted
    107,  # ornamentMordent
    108,  # stringsDownBow
    109,  # stringsDownBow
    110,  # arpeggiato
    111,  # keaboardPedalPed
    112,  # keyboardPedalUp
    113,  # tuplet3
    114,  # tuplet6
    115,  # fingering0
    116,  # fingering1
    117,  # fingering2
    118,  # fingering3
    119,  # fingering4
    120,  # fingering5
    121,  # slur
    123,  # tie
    124,  # restHBar
    125,  # dynamicCrescendoHairpin
    126,  # dynamicDiminuendoHairpin
    129,  # tuplet4
    134,  # tupletBracket
    136,  # ottavaBracket
}
default = 1.0
def is_attribute_an_outlier(cat_stats: dict, cat_thresholds: Dict[str, Tuple[float, float]], cls: str, value: float) -> bool:
    if len(crit_selection) != 0 and cls not in crit_selection:
        return False
    low_thr, high_thr = cat_thresholds.get(cls, (None, None))
    if low_thr is None or high_thr is None:
        return False
    if low_thr > high_thr:  # Inverted threshold
        return high_thr < value < low_thr
    return not (low_thr <= value <= high_thr)

def flag_outlier(obbox: np.ndarray, cat_id: int, stats: dict) -> bool:
    if isinstance(obbox, list):
        obbox = np.array(obbox)
    obbox = np.array(obbox).reshape((4, 2))
    cat_stats = stats[str(cat_id)]
    cat_thrs = get_thresholds(cat_stats)
    # check all attributes
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'area', OBBox.get_area(obbox)):
        return True
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'angle', (OBBox.get_angle(obbox)) % 90 ):
        return True
    l1 = np.linalg.norm(obbox[0] - obbox[1])
    l2 = np.linalg.norm(obbox[1] - obbox[2])
    l1, l2 = min(l1, l2), max(l1, l2)
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'l1', l1):
        return True
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'l2', l2):
        return True
    if is_attribute_an_outlier(cat_stats, cat_thrs, 'edge-ratio', l2 / l1):
        return True
    return False

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
    ax.set_title(f"{thr_cls}: {n_outliers} outliers [{minimum}, {maximum}]\n({low_thr} - {high_thr})")
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
    # data = json.loads(input("JSON: "))
    data = json.loads(args.to_geogebra)
    # data = {"a_bbox":[1218,1047,1261,1125],"o_bbox":[1261,1125,1235.844482421875,1045.1585693359375,1215.930908203125,1051.4327392578125,1241.08642578125,1131.274169921875],"cat_id":["25","157"],"area":284,"img_id":"1550","comments":"instance:#00020d;duration:8*2/3;rel_position:0;"}
    # a_bbox = OBBox.expand_corners(np.array(data['a_bbox'])).reshape((4, 2))
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
            if flag_outlier(o_bbox, cat, stats):
                # print(f"Found outlier '{cat_info[cat]['name']}'({cat})")
                if img_id not in imgs.keys():
                    img_info, _ = dataset.obb.get_img_ann_pair(idxs=None, ids=[int(img_id)])
                    filename = img_info[0]['filename']
                    imgs[img_id] = (osp.join(img_dir, filename), [])
                imgs[img_id][1].append((cat, a_bbox, o_bbox, idx))
    return imgs

def draw_outliers(imgs: dict, cat_info: dict) -> dict:
    outlier_stats = {}
    for img_id, (img_fp, bboxes) in sorted(imgs.items(), reverse=True, key=lambda kvp: len(kvp[1])):
        img = Image.open(img_fp)
        draw = ImageDraw.Draw(img)
        cats = set(map(lambda tpl: tpl[0], bboxes))
        print(f"Visualized {len(bboxes)} outliers in {osp.basename(img_fp)} from the categories: {cats}")
        for cat, a_bbox, bbox, idx in bboxes:
            draw.rectangle(a_bbox, outline='#223CF0', width=3)
            draw.line(bbox + bbox[:2], fill='#F03C22', width=3)
            text = f"[{idx}] {cat_info[cat]['name']}({cat}): {OBBox.get_area(np.array(bbox).reshape((4, 2))):.1f}"
            print(f"  * {text}")
            bbox_np = np.array(bbox).reshape((4, 2))
            position = int(bbox_np.max(axis=0)[0]), int(bbox_np.min(axis=0)[1])
            x1, y1 = ImageFont.load_default().getsize(text)
            x1 += position[0] + 4
            y1 += position[1] + 4
            draw.rectangle((position[0], position[1], x1, y1), fill='#DCDCDC')
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
