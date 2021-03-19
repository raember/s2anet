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
    2: 10.0,  # ledgerLine
    25: 10.0,  # noteheadBlackOnLine
    27: 6.0,  # noteheadBlackInSpace
    31: 10.0,  # noteheadHalfInSpace
    33: 3.0,  # noteheadWholeOnLine
    42: 20.0,  # stem
    64: 3.0,  # accidentialSharp
    70: 3.0,  # keySharp
    88: 2.0,  # rest8th
    90: 2.0,  # rest32nd
    113: 10.0,  # tuplet3
    122: 20.0,  # beam
    135: 5.0,  # staff
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
    35,  # noteheadWholeInSpace
    37,  # noteheadDoubleWholeOnLine
    39,  # noteheadDoubleWholeInSpace
    41,  # augmentationDot
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
    114,  # tuplet6
    115,  # fingering0
    116,  # fingering1
    117,  # fingering2
    118,  # fingering3
    119,  # fingering4
    120,  # fingering5
    121,  # slur
    123,  # tie
    125,  # dynamicCrescendoHairpin
    126,  # dynamicDiminuendoHairpin
    129,  # tuplet4
    134,  # tupletBracket
    136,  # ottavaBracket
}
default = 1.0
def flag_area(stats: dict, area: float) -> bool:
    if stats['id'] in ignore:
        return False
    mean = stats['mean']
    median = stats['median']
    std = stats['std']
    return abs(area - median) > deviation.get(int(stats['id']), default) * std  # or area == 0.0

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
        # if cat_id in {'64'}:
        plot(cat_id, sts)

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
    ann_file='data/deep_scores_dense/deepscores_train.json',
    img_prefix='data/deep_scores_dense/images/',
    pipeline=pipeline,
    use_oriented_bboxes=True,
)
dataset_test = DeepScoresV2Dataset(
    ann_file='data/deep_scores_dense/deepscores_test.json',
    img_prefix='data/deep_scores_dense/images/',
    pipeline=pipeline,
    use_oriented_bboxes=True,
)
cat_info = dataset_train.obb.cat_info

def extract_areas(dataset: DeepScoresV2Dataset):
    areas_by_cat = {}
    for cat_id, o_bbox in zip(dataset.obb.ann_info['cat_id'], dataset.obb.ann_info['o_bbox']):
        cat = int(cat_id[0])
        bbox = np.array(o_bbox).reshape((4, 2))
        area = OBBox.get_area(bbox)
        areas_by_cat[cat] = np.append(areas_by_cat.get(cat, np.array([])), area)
    return areas_by_cat

def extract_stats(cat_to_area: dict, cat_info: dict, stats: dict):
    for cat, area_lst in cat_to_area.items():
        # cat = int(cat_id[0])
        stats[int(cat)] = {
            'id': int(cat),
            'name': cat_info[cat]['name'],
            'min': min(area_lst),
            'max': max(area_lst),
            'mean': np.mean(area_lst),
            'median': np.median(area_lst),
            'std': np.std(area_lst),
            'length': area_lst.size,
            'sorted_areas': sorted(area_lst, reverse=True)
        }

def filter_bboxes(dataset: DeepScoresV2Dataset, stats: dict) -> dict:
    imgs = {}
    img_dir = osp.join(osp.split(dataset.obb.ann_file)[0], 'images')
    for cat_id, o_bbox, img_id, idx in zip(
            dataset.obb.ann_info['cat_id'],
            dataset.obb.ann_info['o_bbox'],
            dataset.obb.ann_info['img_id'],
            dataset.obb.ann_info.index,
    ):
        cat = int(cat_id[0])
        if flag_outlier(o_bbox, cat, stats):
            # print(f"Found outlier '{cat_info[cat]['name']}'({cat})")
            if img_id not in imgs.keys():
                img_info, _ = dataset.obb.get_img_ann_pair(idxs=None, ids=[int(img_id)])
                filename = img_info[0]['filename']
                imgs[img_id] = (osp.join(img_dir, filename), [])
            imgs[img_id][1].append((cat, o_bbox, idx))
    return imgs

def draw_outliers(imgs: dict, cat_info: dict) -> dict:
    outlier_stats = {}
    for img_id, (img_fp, bboxes) in sorted(imgs.items(), reverse=True, key=lambda kvp: len(kvp[1])):
        img = Image.open(img_fp)
        draw = ImageDraw.Draw(img)
        cats = set(map(lambda tpl: tpl[0], bboxes))
        print(f"Visualized {len(bboxes)} outliers in {osp.basename(img_fp)} from the categories: {cats}")
        for cat, bbox, idx in bboxes:
            draw.line(bbox + bbox[:2], fill='#F03C22', width=3)
            text = f"[{idx}] {cat_info[cat]['name']}({cat}): {OBBox.get_area(np.array(bbox).reshape((4, 2)))}"
            print(f"  * {text}")
            position = np.array(bbox).reshape((4, 2)).max(axis=0)
            x1, y1 = ImageFont.load_default().getsize(text)
            x1 += position[0] + 4
            y1 += position[1] + 4
            draw.rectangle((position[0], position[1], x1, y1), fill='#DCDCDC')
            draw.text((position[0] + 2, position[1] + 2), text, '#F03C22')
            outlier_stats[cat] = outlier_stats.get(cat, 0) + 1
        img.save(osp.join('out_debug', osp.basename(img_fp)))
    return outlier_stats

if args.compile:
    stats = {}
    print("Gathering the stats")
    cat_to_area_train = extract_areas(dataset_train)
    cat_to_area_test = extract_areas(dataset_test)
    print("Calculating the stats")
    extract_stats(cat_to_area_train, cat_info, stats)
    extract_stats(cat_to_area_test, cat_info, stats)
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
    imgs_train = filter_bboxes(dataset_train, stats)
    imgs_test = filter_bboxes(dataset_test, stats)
    print("Drawing outliers and saving them to disk")
    outlier_stats_train = draw_outliers(imgs_train, cat_info)
    print(f"{'#'*10} TEST DATASET {'#'*10}")
    outlier_stats_test = draw_outliers(imgs_test, cat_info)
    print("Train stats:")
    for cat, number in sorted(outlier_stats_train.items(), reverse=True, key=lambda kvp: kvp[1]):
        print(f"{cat_info[cat]['name']} ({cat}): {number}")
    print("Test stats:")
    for cat, number in sorted(outlier_stats_test.items(), reverse=True, key=lambda kvp: kvp[1]):
        print(f"{cat_info[cat]['name']} ({cat}): {number}")