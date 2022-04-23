import json
from argparse import ArgumentParser
from collections import defaultdict

from PIL import Image, ImageDraw
from PIL import ImageFont
from typing import Tuple, List, Dict, Optional

from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from pandas import DataFrame, Series

from mmdet.datasets import DeepScoresV2Dataset
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt, axes

from mmdet.datasets.deepscoresv2 import threshold_classes, get_thresholds
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

cat_selection = {}
if len(cat_selection) == 0:
    cat_selection = set(map(str, range(1, 137)))
crit_selection = {}
if len(crit_selection) == 0:
    crit_selection = {*threshold_classes}


def is_attribute_an_outlier(cat_thresholds: Dict[str, Tuple[float, float]], cls: str, value: float) -> bool:
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
    if is_attribute_an_outlier(cat_thrs, 'area', area):
        reasons['area'] = area
    angle = (OBBox.get_angle(obbox)) % 90
    if is_attribute_an_outlier(cat_thrs, 'angle', angle):
        reasons['angle'] = angle
    l1 = np.linalg.norm(obbox[0] - obbox[1])
    l2 = np.linalg.norm(obbox[1] - obbox[2])
    l1, l2 = min(l1, l2), max(l1, l2)
    if is_attribute_an_outlier(cat_thrs, 'l1', l1):
        reasons['l1'] = l1
    if is_attribute_an_outlier(cat_thrs, 'l2', l2):
        reasons['l2'] = l2
    if is_attribute_an_outlier(cat_thrs, 'edge-ratio', l2 / l1):
        reasons['edge-ratio'] = l2 / l1
    return reasons

def plot_attribute(ax: Axes, cat_stats: dict, cat_thrs: dict, thr_cls: str):
    cls_stats = cat_stats[thr_cls]
    values = sorted(cls_stats['values'], reverse=True)
    n = len(values)
    n_outliers = sum(map(lambda value: is_attribute_an_outlier(cat_thrs, thr_cls, value), values))
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
    fig.suptitle(f"[{cat_id}] {cat_stats['name']}: {len(cat_stats[threshold_classes[0]]['values'])}")
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
            'values': values
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
    def map_row(x: Series) -> Series:
        abbox, obbox, cls_id = x['a_bbox'], x['o_bbox'], x['cat_id']
        # make abbox into a 8-tuple like obbox
        abbox = [abbox[2], abbox[3], abbox[2], abbox[1], abbox[0], abbox[1], abbox[0], abbox[3]]
        abbox = fix_ann(abbox)
        x['a_bbox'] = [abbox[4], abbox[5], abbox[0], abbox[1]]
        if '81' in cls_id or '82' in cls_id:  # Align obbox to abbox for fermata
            obbox = abbox
        else:
            obbox = fix_ann(obbox)
        x['o_bbox'] = obbox
        x['area'] = int(OBBox.get_area(np.array([obbox[0::2], obbox[1::2]]).T))
        return x
    stem_min_l2 = get_thresholds(stats['42'])['l2'][0]
    def flag_row(x: Series) -> bool:
        abbox, cls_id = x['a_bbox'], x['cat_id']
        if '42' in cls_id:  # delete stems that are too short (likely because of overlapping stems)
            l2 = max(abs(abbox[0] - abbox[2]), abs(abbox[1] - abbox[3]))
            return l2 < stem_min_l2
        return False
    # Fix bboxes
    anns.ann_info = anns.ann_info.apply(map_row, axis=1)
    # Delete bboxes
    to_delete = anns.ann_info.apply(flag_row, axis=1)
    to_delete |= anns.ann_info.index.isin([
        # vanished clef 8
        670134, 670132, 670186, 670147, 670142,
        # faulty slur
        346823,
    ])
    to_del_anns = anns.ann_info[to_delete]
    anns.ann_info.drop(to_del_anns.index, inplace=True, errors='ignore')
    ann_ids = set(map(str, to_del_anns.index))
    if len(ann_ids) > 0:
        for im in anns.img_info:
            for ann_id in ann_ids.intersection(im['ann_ids']):
                im['ann_ids'].remove(ann_id)
    else:
        print('No annotations to delete')
    print()

if args.fix_annotations:
    if 'stats' not in globals():
        with open(STAT_FILE, 'r') as fp:
            stats = json.load(fp)
    print('[TRAIN] Fixing annotations')
    fix_annotations(dataset_train.obb)
    print('[TRAIN] Saving dataset to deepscores_train.fixed.json')
    dataset_train.obb.save_annotations('deepscores_train.fixed.json')
    print('[TEST] Fixing annotations')
    fix_annotations(dataset_test.obb)
    print('[TEST] Saving dataset to deepscores_test.fixed.json')
    dataset_test.obb.save_annotations('deepscores_test.fixed.json')
    print('Done')
