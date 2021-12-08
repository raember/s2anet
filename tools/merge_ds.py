import shutil

import json
import numpy as np
from obb_anns import OBBAnns
from pathlib import Path

ann_in_1 = OBBAnns('data/deep_scores_dense/deepscores_test.json')
ann_in_2 = OBBAnns('../ili_subset/scores.json')
SUFFIX = '_scn'
ann_out_path = Path('..', 'ds_test_ili', 'deepscores_test.json')

ann_in_1.load_annotations()
ann_in_2.load_annotations()
target_ds_root = ann_out_path.parent

assert target_ds_root.exists(), f"Directory of the target dataset ({str(target_ds_root)}) must exist!"

# Make dirs
images_path = Path('images')
segmentation_path = Path('segmentation')
instance_path = Path('instance')
for subdir in [images_path, segmentation_path, instance_path]:
    (target_ds_root / subdir).mkdir(exist_ok=True)

print("Copying image files")
def copy_images(ann: OBBAnns, suffix: str = ''):
    ds_root = Path(ann.ann_file).parent
    for img_info in ann.img_info:
        img_path = Path(img_info['filename'])
        img_path_new = img_path.with_name(img_path.with_suffix('').name + suffix + img_path.suffix)
        for subdir, suff in [(images_path, ''), (segmentation_path, '_seg'), (instance_path, '_inst')]:
            img_path_sub = img_path.with_name(img_path.with_suffix('').name + suff + img_path.suffix)
            img_path_sub_new = img_path_new.with_name(img_path_new.with_suffix('').name + suff + img_path_new.suffix)
            source_path = ds_root / subdir / img_path_sub
            target_path = target_ds_root / subdir / img_path_sub_new
            shutil.copy(source_path, target_path)
print(1)
copy_images(ann_in_1)
print(2)
copy_images(ann_in_2, SUFFIX)

print("Compile annotation data")
new_ann = {'info': {
    'description': 'Deepscores test set with scanned samples in the OBB format',
    'version': '1.0',
    'year': '2021',
    'contributor': 'Lukas Tuggener, Ismail Elezi, Yvan Satyawan, Jürgen Schmidhuber, Marcello Pelillo, Thilo Stadelmann, Raphael Emberger',
    'date_created': '2021-07-19',
    'url': 'https://tuggeluk.github.io/deepscores/',
}, 'annotation_sets': [
    'deepscores',
    'muscima++',
], 'categories': ann_in_1.cat_info, 'images': [],
    'annotations': (ann_in_1.ann_info + ann_in_2.ann_info).to_dict('index')
}
img_info = ann_in_1.img_info
ann_info = ann_in_1.ann_info + ann_in_2.ann_info
ann_id = ann_in_1.ann_info.shape[0]

print("Compile image data")
for _img_info in ann_in_2.img_info:
    file = Path(_img_info['filename'])
    file = file.with_name(file.with_suffix('').name + SUFFIX + file.suffix)
    _img_info['filename'] = str(file)
    ann_ids = np.array(list(map(int, _img_info['ann_ids'])))
    ann_ids += ann_id
    _img_info['ann_ids'] = list(map(str, ann_ids))
    img_info.append(_img_info)
new_ann['images'] = img_info

print("Writing to disk")
with open(str(target_ds_root / Path('scores.json')), 'w') as fp:
    json.dump(new_ann, fp)
