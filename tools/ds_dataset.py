import csv
import json
import math
from pathlib import Path

import numpy
from PIL import Image

from mmdet.core import rotated_box_to_poly_single
from obb_anns import OBBAnns

MAIN_ANN_FILE = 'data/deep_scores_dense/imslp_test.json'
DELETE_ENTRIES = False
NEW_ANNS = {
    # 'in/Liszt_Franz_Konzertetude.png': 'in/Liszt_Franz_Konzertetude_Annotations.csv',
    'in/The_Entertainer#1.png': 'in/The_Entertainer#1.V2.json',
    'in/The_Entertainer#2.png': 'in/The_Entertainer#2.V2.json',
    'in/The_Entertainer#3.png': 'in/The_Entertainer#3.V2.json',
    'in/The_Entertainer#4.png': 'in/The_Entertainer#4.V2.json',
}
ANN_OUT_FILE = Path('data/deep_scores_dense/imslp_test_new.json')
WIDTH = 1960.0

if __name__ == '__main__':
    anns = OBBAnns(MAIN_ANN_FILE)
    anns.load_annotations()
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
        next_img_id = max(anns.img_idx_lookup.keys()) + 1
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
                        'img_id': next_img_id,
                        'comments': ''
                    }
        elif ann_file.suffix == '.json':
            data = json.load(open(ann_file, 'r'))
            for bbox in data['bounding_boxes']:
                rect = bbox[:5]
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
                    'img_id': next_img_id,
                    'comments': ''
                }
        anns.add_new_img_ann_pair(img_file, img.width, img.height, new_anns)
        img.save(ANN_OUT_FILE.parent / 'images' / Path(img_file).name)
    anns.save_annotations(ANN_OUT_FILE)
