import json
from pathlib import Path

import pandas as pd
from obb_anns import OBBAnns

orig_ann_root = Path('..', 'data', 'deep_scores_dense')

from_ann = orig_ann_root / 'deepscores_test.json'
to_ann = orig_ann_root / 'deepscores_test_sage_test.json'

N_SAMPLES = 200  # set to None to keep original size

KEEP_CLASSES = None  # add index of all classes to keep, set to None to keep all classes


def get_annotations() -> (OBBAnns, pd.DataFrame):
    ann = OBBAnns(str(from_ann))
    ann.load_annotations()
    old_ann = ann.ann_info.copy(deep=True)

    return ann, old_ann


def reduce_ann_size(ann: OBBAnns, old_ann: pd.DataFrame) -> OBBAnns:
    imgs = []
    anns = []
    for _ in range(N_SAMPLES):
        img_info = ann.img_info.pop(0)
        anns += img_info['ann_ids']
        imgs.append(img_info)

    ann.ann_info = old_ann.loc[list(map(int, anns)), :]
    ann.img_info = imgs
    return ann


def select_classes(ann: OBBAnns) -> OBBAnns:
    ann.cat_info = {k: ann.cat_info[k] for k in KEEP_CLASSES}

    # create a Series containing all annotations to keep
    keep_index = ann.ann_info.index.to_series()
    keep_index[:] = False

    for c in KEEP_CLASSES:
        keep_index = keep_index | (ann.ann_info.cat_id.apply(lambda x: str(c) in x))

    # remove annotation infos
    ann.ann_info = ann.ann_info[keep_index]

    # remove image info
    img_to_keep = keep_index[keep_index].index.values
    for img in ann.img_info:
        img['ann_ids'] = [ann_id for ann_id in img['ann_ids'] if int(ann_id) in img_to_keep]
        if len(img['ann_ids']) <= 0:
            ann.img_info.remove(img)

    return ann


def obb_anns_to_json(ann: OBBAnns) -> dict:
    with open(ann.ann_file, 'r') as ann_file:
        data = json.load(ann_file)
    data['images'] = ann.img_info
    data['annotations'] = {str(key): value for key, value in ann.ann_info.to_dict('index').items()}
    return data


def main():
    ann, old_ann = get_annotations()
    if N_SAMPLES is not None:
        ann = reduce_ann_size(ann, old_ann)

    if KEEP_CLASSES is not None:
        ann = select_classes(ann)

    data = obb_anns_to_json(ann)
    with open(to_ann, 'w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    main()
