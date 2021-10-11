import json
from argparse import ArgumentParser

from PIL import Image, ImageColor, ImageDraw, ImageFont
from PIL.ImageFont import ImageFont

from mmdet.datasets import DeepScoresV2Dataset
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt

from mmdet.datasets.pipelines.transforms import OBBox

parser = ArgumentParser(description='Deepscores statistics')
parser.add_argument('-c', '--compile', dest='compile', action='store_true', default=False,
                    help='Compiles statistics')
parser.add_argument('-f', '--flag-outliers', dest='flag_outliers', action='store_true', default=False,
                    help='Flags outliers using past statistics')
args = parser.parse_args()

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
# def print_text_label(draw, position, text, color_text, color_box):
#     x1, y1 = ImageFont.load_default().getsize(text)
#     x1 += position[0] + 4
#     y1 += position[1] + 4
#     draw.rectangle((position[0], position[1], x1, y1), fill=color_box)
#     draw.text((position[0] + 2, position[1] + 2), text, color_text)
#     return draw, x1, position[1]
STAT_FILE = 'stats.json'
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
        stats[cat] = {
            'name': dataset.cat2label[cat],
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
    for cat_id, o_bbox, img_id in zip(dataset.obb.ann_info['cat_id'], dataset.obb.ann_info['o_bbox'], dataset.obb.ann_info['img_id']):
        cat = int(cat_id[0])
        # bracket: 1
        # ledgerLine: 2
        # stem: 42
        # slur: 121
        # beam: 122
        # tie: 123
        # dynamicCrescendoHairpin: 125
        # dynamicDiminuendoHairpin: 126
        # tupletBracket: 134
        # staff: 135
        # ottavaBracket: 136
        img_id = int(img_id)
        bbox = np.array(o_bbox).reshape((4, 2))
        area = OBBox.get_area(bbox)
        mean = stats[str(cat)]['mean']
        median = stats[str(cat)]['median']
        std = stats[str(cat)]['std']
        if (area - median) > (3.5 if cat not in {
            1, 2, 42, 121, 122, 123, 125, 126, 134, 135, 136
        } else 12.0) * std:  # or area == 0.0:
            print(f"Found outlier '{dataset.obb.cat_info[cat]['name']}'({cat}): {area:.8} is too far away from mean {mean:.8}")
            if img_id not in imgs.keys():
                imgs[img_id] = []
            imgs[img_id].append((cat, bbox))
    img_dir = osp.join(osp.split(dataset.obb.ann_file)[0], 'images')
    print("Drawing outliers and saving them to disk")
    outlier_stats = {}
    for img_id, bboxes in sorted(imgs.items(), reverse=True, key=lambda kvp: kvp[1].size):
        img_info, _ = dataset.obb.get_img_ann_pair(idxs=None, ids=[img_id])
        filename = img_info[0]['filename']
        img_fp = osp.join(img_dir, filename)
        img = Image.open(img_fp)
        draw = ImageDraw.Draw(img)
        cats = set(map(lambda tpl: tpl[0], bboxes))
        for cat, npbbox in bboxes:
            bbox = list(npbbox.reshape((8,)))
            draw.line(bbox + bbox[:2], fill='#F03C22', width=3)
            outlier_stats[cat] = outlier_stats.get(cat, 0) + 1
        img.save(osp.join('out_debug', filename))
        print(f"Visualized {len(bboxes)} outliers in {osp.basename(filename)} from the categories: {cats}")
    for cat, number in sorted(outlier_stats.items(), reverse=True, key=lambda kvp: kvp[1]):
        print(f"{dataset.obb.cat_info[cat]['name']} ({cat}): {number}")
    print("Done")