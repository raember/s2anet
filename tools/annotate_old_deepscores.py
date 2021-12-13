import json

import shutil

from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from obb_anns import OBBAnns
from pathlib import Path

# source_images = Path('..', 'ili_subset', 'images')
# source_images = Path('..', 'scanned_deepscore_images_png')
source_images = Path('..', 'scanned_deepscore_images_clean')
# source_index = Path('..', 'clean_scans')
# source_index = Path('..', 'scanned_deepscore_images_png')
source_index = Path('..', 'scanned_deepscore_images_clean')
# target_ds = Path('..', 'ili_subset')
target_ds = Path('..', 'deepscores_scanned')
target_ds_file = target_ds / 'deepscores.json'

orig_ann_root =  Path('data', 'deep_scores_dense')
orig_ann_train = OBBAnns(str(orig_ann_root / 'deepscores_train.json'))
orig_ann_train.load_annotations()
orig_ann_test = OBBAnns(str(orig_ann_root / 'deepscores_test.json'))
orig_ann_test.load_annotations()

def construct_reverse_lookup(ann: OBBAnns) -> dict:
    reverse_lookup = {}
    for entry in ann.img_info:
        reverse_lookup[entry['filename']] = entry
    return reverse_lookup

reverse_lookup_train = construct_reverse_lookup(orig_ann_train)
reverse_lookup_test = construct_reverse_lookup(orig_ann_test)

new_ann = {'info': {
    'description': 'Deepscores set of scans in the OBB format',
    'version': '1.0',
    'year': '2021',
    'contributor': 'Lukas Tuggener, Ismail Elezi, Yvan Satyawan, Jürgen Schmidhuber, Marcello Pelillo, Thilo Stadelmann, Raphael Emberger',
    'date_created': '2021-07-28',
    'url': 'https://tuggeluk.github.io/deepscores/',
}, 'annotation_sets': [
    'deepscores',
    'muscima++',
], 'categories': orig_ann_train.cat_info, 'images': [], 'annotations': {}}
img_id = 0
ann_id = 0
print("Creating annotation lookup (train)")
anns_train = orig_ann_train.ann_info.to_dict('index')
print("Creating annotation lookup (test)")
anns_test = orig_ann_test.ann_info.to_dict('index')
for file in source_index.glob('*.png'):
    # Determine which dataset to source from
    file = source_images / file.name
    entry = reverse_lookup_test.get(file.name, None)
    orig_data = orig_ann_test
    orig_anns = anns_test
    if entry is None:
        entry = reverse_lookup_train.get(file.name, None)
        orig_data = orig_ann_train
        orig_anns = anns_train

    # If we don't find an entry, remove image
    if entry is None:
        print(f"Entry not found: {file.name}")
        continue

    print(f"Processing: {file.name}")
    orig: PngImageFile = Image.open(file)
    img = orig.resize((entry['width'], entry['height']), Image.NEAREST)
    img.save(target_ds / 'images' / file.name)
    seg_name = file.with_name(f"{file.with_suffix('').name}_seg.png").name
    seg = Path('data', 'deep_scores_dense', 'segmentation', seg_name)
    shutil.copy(seg, target_ds / 'segmentation' / seg_name)
    ins_name = file.with_name(f"{file.with_suffix('').name}_inst.png").name
    ins = Path('data', 'deep_scores_dense', 'instance', ins_name)
    shutil.copy(ins, target_ds / 'instance' / ins_name)
    # continue
    img_id += 1
    ann_ids = []
    for old_ann_id in entry['ann_ids']:
        ann_id += 1
        # a_bbox, o_bbox, cat_id, area, _, comments = df.loc[df['img_id'] == entry['id']].values[0]
        new_ann['annotations'][str(ann_id)] = orig_anns[int(old_ann_id)]
        new_ann['annotations'][str(ann_id)]['img_id'] = str(img_id)
        ann_ids.append(ann_id)
    entry['ann_ids'] = ann_ids
    entry['id'] = img_id
    new_ann['images'].append(entry)
with open(target_ds_file, 'w') as fp:
    json.dump(new_ann, fp)
print(f"Processed {len(new_ann['images'])}")
