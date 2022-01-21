import json

from obb_anns import OBBAnns
from pathlib import Path


orig_ann_root =  Path('..', 'data', 'deep_scores_dense')

from_ann = orig_ann_root / 'deepscores_test.json'
to_ann = orig_ann_root / 'deepscores_test_edited.json'

N_SAMPLES = 200

ann = OBBAnns(str(from_ann))
ann.load_annotations()

old_anns = ann.ann_info.copy(deep=True)
old_anns

imgs = []
anns = []
for _ in range(N_SAMPLES):
    img_info = ann.img_info.pop(0)
    anns += img_info['ann_ids']
    imgs.append(img_info)

ann.ann_info = old_anns.loc[list(map(int, anns)), :]
ann.img_info = imgs

def obb_anns_to_json(obb: OBBAnns) -> dict:
    with open(obb.ann_file, 'r') as ann_file:
        data = json.load(ann_file)
    data['images'] = ann.img_info
    data['annotations'] = {str(key):value for key, value in ann.ann_info.to_dict('index').items()}
    return data

data = obb_anns_to_json(ann)
with open(to_ann, 'w') as fp:
    json.dump(data, fp)
