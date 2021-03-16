import json
from argparse import ArgumentParser

from PIL import Image, ImageDraw
from PIL import ImageFont

from mmdet.datasets import DeepScoresV2Dataset
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt

from mmdet.datasets.pipelines.transforms import OBBox

parser = ArgumentParser(description='Deepscores statistics')
parser.add_argument('-c', '--compile', dest='compile', action='store_true', default=False,
                    help='Compiles statistics')
parser.add_argument('-p', '--plot-stats', dest='plot_stats', action='store_true', default=False,
                    help='Plots statistics')
parser.add_argument('-f', '--flag-outliers', dest='flag_outliers', action='store_true', default=False,
                    help='Flags outliers using past statistics')
args = parser.parse_args()

deviation = {
    1: 3.0,  # brace
    2: 5.0,  # ledgerLine
    6: 2.5,  # clefG
    25: 4.0,  # noteheadBlackOnLine
    27: 4.0,  # noteheadBlackInSpace
    29: 3.5,  # noteheadHalfOnLine
    31: 3.0,  # noteheadHalfInSpace
    33: 3.0,  # noteheadWholeOnLine
    35: 2.0,  # noteheadWholeInSpace
    37: 2.0,  # noteheadDoubleWholeOnLine
    39: 2.0,  # noteheadDoubleWholeInSpace
    41: 2.1,  # augmentationDot
    42: 6.0,  # stem
    60: 2.5,  # accidentialFlat
    64: 2.0,  # accidentialSharp
    70: 3.0,  # keySharp
    86: 1.5,  # restHalf
    88: 2.0,  # rest8th
    90: 2.0,  # rest32nd
    110: 2.0,  # arpeggiato
    113: 3.0,  # tuplet3
    118: 3.0,  # fingering3
    119: 3.0,  # fingering4
    121: 5.0,  # slur
    122: 5.0,  # beam
    123: 5.0,  # tie
    125: 3.0,  # dynamicCrescendoHairpin
    126: 3.0,  # dynamicDiminuendoHairpin
    134: 3.0,  # tupletBracket
    135: 3.5,  # staff
}
ignore = {
    3,  # repeatDot
    7,  # clefCAlto
    8,  # clefCTenor
    13,  # timeSig0
    14,  # timeSig1
    15,  # timeSig2
    16,  # timeSig3
    17,  # timeSig4
    19,  # timeSig6
    20,  # timeSig7
    21,  # timeSig8
    22,  # timeSig9
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
    112,  # keyboardPedalUp
    115,  # fingering0
    116,  # fingering1
    117,  # fingering2
    120,  # fingering5
}
default = 1.0
def flag_area(stats: dict, area: float) -> bool:
    if stats['id'] in ignore:
        return False
    mean = stats['mean']
    median = stats['median']
    std = stats['std']
    return (area - median) < deviation.get(int(stats['id']), default) * std  # or area == 0.0

def flag_outlier(bbox: np.ndarray, cat_id: int, stats: dict) -> bool:
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    return flag_area(stats[str(cat_id)], OBBox.get_area(bbox.reshape((4, 2))))

def plot(cat_id: int, stats: dict):
    areas = stats['sorted_areas']
    n = len(areas)
    n_outliers = sum(map(lambda area: flag_area(stats, area), areas))
    plt.title(f"{{{cat_id}}} {stats['name']}: {n_outliers} outliers (std factor: {deviation.get(int(cat_id), default)})")
    plt.xlabel('Index of sorted Areas')
    plt.ylabel('Areas, sorted')
    # plt.yscale('log')
    # plt.ylim(ymin=0, ymax=max(areas))
    plt.ylim(ymin=min(areas), ymax=max(areas))
    plt.grid(True)
    plt.plot(range(n)[::-1], areas)
    median = stats['median']
    mean = stats['mean']
    std = stats['std']
    std_factor = deviation.get(int(cat_id), default)
    indiv_std = std_factor * std
    plt.plot([0, n], [median, median], color='#20dd50')
    plt.plot([0, n], [mean, mean], color='#55ffaa')
    plt.plot([0, n], [median + std, median + std], color='#ff9050')
    plt.plot([0, n], [median - std, median - std], color='#ff9050')
    plt.plot([0, n], [median + indiv_std, median + indiv_std], color='#dd4020')
    plt.plot([0, n], [median - indiv_std, median - indiv_std], color='#dd4020')
    plt.show()

STAT_FILE = 'stats.json'
if args.plot_stats:
    print("Loading stats")
    if 'stats' not in globals():
        with open(STAT_FILE, 'r') as fp:
            stats = json.load(fp)
    for cat_id, sts in sorted(stats.items(), key=lambda kvp: int(kvp[0])):
        if cat_id in {'41'}:
            plot(cat_id, sts)


dataset = DeepScoresV2Dataset(
    ann_file='data/deep_scores_dense/deepscores_train.json',
    img_prefix='data/deep_scores_dense/images/',
    pipeline=[
        {'type': 'LoadImageFromFile'},
        {'type': 'LoadAnnotations', 'with_bbox': True},
        {'type': 'RandomCrop', 'crop_size': (1400, 1400), 'threshold_rel': 0.6, 'threshold_abs': 20.0},
        {'type': 'RotatedResize', 'img_scale': (1024, 1024), 'keep_ratio': True},
        {'type': 'RotatedRandomFlip', 'flip_ratio': 0.0},
        {'type': 'Normalize', 'mean': [240, 240, 240], 'std': [57, 57, 57], 'to_rgb': False},
        {'type': 'Pad', 'size_divisor': 32},
        {'type': 'DefaultFormatBundle'},
        {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}
    ],
    use_oriented_bboxes=True,
)

if args.compile:
    stats = {}
    areas_by_cat = {}
    print("Gathering the stats")
    for cat_id, o_bbox in zip(dataset.obb.ann_info['cat_id'], dataset.obb.ann_info['o_bbox']):
        cat = int(cat_id[0])
        bbox = np.array(o_bbox).reshape((4, 2))
        area = OBBox.get_area(bbox)
        areas_by_cat[cat] = np.append(areas_by_cat.get(cat, np.array([])), area)
    print("Calculating the stats")
    for cat, areas in areas_by_cat.items():
        # cat = int(cat_id[0])
        stats[int(cat)] = {
            'id': int(cat),
            'name': dataset.obb.cat_info[cat]['name'],
            'min': min(areas),
            'max': max(areas),
            'mean': np.mean(areas),
            'median': np.median(areas),
            'std': np.std(areas),
            'length': areas.size,
            'sorted_areas': sorted(areas, reverse=True)
        }
    print("Saving the stats")
    with open(STAT_FILE, 'w') as fp:
        json.dump(stats, fp, indent=4)
    print("Done")

if args.flag_outliers:
    print("Loading stats")
    if 'stats' not in globals():
        with open(STAT_FILE, 'r') as fp:
            stats = json.load(fp)
    imgs = {}
    print("Searching for outliers")
    for cat_id, o_bbox, img_id, idx in zip(
            dataset.obb.ann_info['cat_id'],
            dataset.obb.ann_info['o_bbox'],
            dataset.obb.ann_info['img_id'],
            dataset.obb.ann_info.index,
    ):
        cat = int(cat_id[0])
        if flag_outlier(o_bbox, cat, stats):
            # print(f"Found outlier '{dataset.obb.cat_info[cat]['name']}'({cat})")
            if img_id not in imgs.keys():
                imgs[img_id] = []
            imgs[img_id].append((cat, o_bbox, idx))
    img_dir = osp.join(osp.split(dataset.obb.ann_file)[0], 'images')
    print("Drawing outliers and saving them to disk")
    outlier_stats = {}
    for img_id, bboxes in sorted(imgs.items(), reverse=True, key=lambda kvp: len(kvp[1])):
        img_info, _ = dataset.obb.get_img_ann_pair(idxs=None, ids=[int(img_id)])
        filename = img_info[0]['filename']
        img_fp = osp.join(img_dir, filename)
        img = Image.open(img_fp)
        draw = ImageDraw.Draw(img)
        cats = set(map(lambda tpl: tpl[0], bboxes))
        for cat, bbox, idx in bboxes:
            draw.line(bbox + bbox[:2], fill='#F03C22', width=3)
            text = f"{{{cat}}} {dataset.obb.cat_info[cat]['name']} [{idx}]"
            position = np.array(bbox).reshape((4, 2)).max(axis=0)
            x1, y1 = ImageFont.load_default().getsize(text)
            x1 += position[0] + 4
            y1 += position[1] + 4
            draw.rectangle((position[0], position[1], x1, y1), fill='#DCDCDC')
            draw.text((position[0] + 2, position[1] + 2), text, '#F03C22')
            outlier_stats[cat] = outlier_stats.get(cat, 0) + 1
        img.save(osp.join('out_debug', filename))
        print(f"Visualized {len(bboxes)} outliers in {osp.basename(filename)} from the categories: {cats}")
    for cat, number in sorted(outlier_stats.items(), reverse=True, key=lambda kvp: kvp[1]):
        print(f"{dataset.obb.cat_info[cat]['name']} ({cat}): {number}")
    print("Done")