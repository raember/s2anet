import csv
import math

import pandas
from PIL import Image

from mmdet.core import rotated_box_to_poly_single
from obb_anns import OBBAnns

MAIN_ANN_FILE = 'data/deep_scores_dense/imslp_test.json'
DELETE_ENTRIES = False
NEW_ANNS = {
    'Liszt_Franz_Konzertetude.png': 'Liszt_Franz_Konzertetude_Annotations.csv'
}
ANN_OUT_FILE = 'data/deep_scores_dense/imslp_test_new.json'

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
        img = img.resize((1960, int(1960.0 / img.width * img.height)), Image.NEAREST)
        with open(ann_file, 'r') as csvfile:
            new_anns = {}
            next_img_id = max(anns.img_idx_lookup.keys()) + 1
            for row in csv.reader(csvfile):
                cls_name = row[0]
                rects = list(map(int, row[1:]))
                rects[-1] = rects[-1] / 180.0 * math.pi
                o_bbox = rotated_box_to_poly_single(rects)
                a_bbox = list(map(int, [max(o_bbox[0::2]), min(o_bbox[1::2]), min(o_bbox[0::2]), max(o_bbox[1::2])]))
                area = rects[2] * rects[3]
                cat_id = classes[cls_name.lower()]
                anns_id += 1
                new_anns[anns_id] = {
                    'a_bbox': a_bbox,
                    'o_bbox': o_bbox.tolist(),
                    'cat_id': [str(cat_id)],
                    'area': area,
                    'img_id': next_img_id,
                    'comments': ''
                }
            anns.add_new_img_ann_pair(img_file, img.width, img.height, new_anns)

    anns.save_annotations(ANN_OUT_FILE)
