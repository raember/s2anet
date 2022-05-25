import csv
import json
import math
import os
import tarfile
from pathlib import Path

import numpy
from PIL import Image

from mmdet.core import rotated_box_to_poly_single
from obb_anns import OBBAnns

MAIN_ANN_FILE = Path('data/deep_scores_dense/imslp_test.json')
DELETE_ENTRIES = False
VISUALIZE = True
CREATE_TAR = True
NEW_ANNS = {
    # 'in/Liszt_Franz_Konzertetude.png': 'in/Liszt_Franz_Konzertetude_Annotations.csv',
    'in/The_Entertainer#1.png': 'in/The_Entertainer#1.V2.json',
    'in/The_Entertainer#2.png': 'in/The_Entertainer#2.V2.json',
    'in/The_Entertainer#3.png': 'in/The_Entertainer#3.V2.json',
    'in/The_Entertainer#4.png': 'in/The_Entertainer#4.V2.json',
    'in/grieg-peer-gynt-suite-no-1-op-46-piano#2.png': 'in/grieg-peer-gynt-suite-no-1-op-46-piano#2.V2.json',
    'in/grieg-peer-gynt-suite-no-1-op-46-piano#3.png': 'in/grieg-peer-gynt-suite-no-1-op-46-piano#3.V2.json',
    'in/IMSLP529977#4.png': 'in/IMSLP529977#4.V2.json',
}
ANN_OUT_FILE = Path('data/deep_scores_dense/imslp_test_new.json')
VISUALIZATION_OUT_FOLDER = Path(f"out_{MAIN_ANN_FILE.stem}")
WIDTH = 1960.0

if __name__ == '__main__':
    anns = OBBAnns(MAIN_ANN_FILE)
    anns.load_annotations()
    if DELETE_ENTRIES:
        anns.ann_info = anns.ann_info.drop(anns.ann_info.index)
        anns.img_info.clear()
        anns.img_idx_lookup.clear()
    classes = {}
    for cat_id, data in anns.cat_info.items():
        if data['annotation_set'] != 'deepscores':
            continue
        classes[data['name'].lower()] = cat_id

    anns_id = anns.ann_info.shape[0]
    for img_file, ann_file in NEW_ANNS.items():
        img = Image.open(img_file)
        scale = WIDTH / img.width
        img = img.resize((int(WIDTH), int(scale * img.height)), Image.NEAREST)
        ann_file = Path(ann_file)
        assert ann_file.exists()
        new_anns = {}
        img_id = max([*anns.img_idx_lookup.keys(), 0]) + 1
        if ann_file.suffix == '.csv':
            with open(ann_file, 'r') as csvfile:
                for row in csv.reader(csvfile):
                    cls_name = row[0]
                    rect = [int(scale * float(val)) for val in row[1:5]]
                    angle = float(row[5])
                    rect.append(angle / 180.0 * math.pi)
                    o_bbox = rotated_box_to_poly_single(rect)
                    a_bbox = list(map(int, [max(o_bbox[0::2]), min(o_bbox[1::2]), min(o_bbox[0::2]), max(o_bbox[1::2])]))
                    area = rect[2] * rect[3]
                    cat_id = classes[cls_name.lower()]
                    anns_id += 1
                    new_anns[anns_id] = {
                        'a_bbox': a_bbox,
                        'o_bbox': o_bbox.astype(int).tolist(),
                        'cat_id': [str(cat_id)],
                        'area': area,
                        'img_id': img_id,
                        'comments': ''
                    }
        elif ann_file.suffix == '.json':
            data = json.load(open(ann_file, 'r'))
            for bbox in data['bounding_boxes']:
                rect = [int(scale * float(val)) for val in bbox[:4]]
                rect[0] += rect[2] // 2
                rect[1] += rect[3] // 2
                rect.append(bbox[4])
                cls_name = bbox[5]
                o_bbox = rotated_box_to_poly_single(rect)
                a_bbox = list(map(int, [max(o_bbox[0::2]), min(o_bbox[1::2]), min(o_bbox[0::2]), max(o_bbox[1::2])]))
                area = rect[2] * rect[3]
                cat_id = classes[cls_name.lower()]
                anns_id += 1
                new_anns[anns_id] = {
                    'a_bbox': a_bbox,
                    'o_bbox': o_bbox.astype(int).tolist(),
                    'cat_id': [str(cat_id)],
                    'area': area,
                    'img_id': img_id,
                    'comments': ''
                }
        anns.add_new_img_ann_pair(img_file, img.width, img.height, new_anns)
        img.save(ANN_OUT_FILE.parent / 'images' / Path(img_file).name)
        if VISUALIZE:
            anns.visualize(img_id=img_id, data_root=str(MAIN_ANN_FILE.parent), out_dir=str(VISUALIZATION_OUT_FOLDER))
    anns.save_annotations(ANN_OUT_FILE)
    if CREATE_TAR:
        os.chdir(MAIN_ANN_FILE.parent)
        with tarfile.open(MAIN_ANN_FILE.with_suffix('.tar').name, 'w:') as tar:
            tar.add(ANN_OUT_FILE.name, arcname=MAIN_ANN_FILE.name)
            for img_info in anns.img_info:
                tar.add(f"images/{img_info['filename']}")
