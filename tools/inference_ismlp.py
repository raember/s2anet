from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import cv2
from mmdet.core import rotated_box_to_poly_np
from mmcv.visualization import imshow_det_bboxes
import numpy as np
from obb_anns import OBBAnns

annotations_file = "data/deep_scores_dense/deepscores_test.json"
obb = OBBAnns(annotations_file)
obb.load_annotations()
obb.set_annotation_set_filter(['deepscores'])
CLASSES = tuple([v["name"] for (k, v) in obb.get_cats().items()])

config_file = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_lowrez.py'
checkpoint_file = 'work_dirs/s2anet_r50_fpn_1x_deepscoresv2_tugg_lowrez/epoch_500.pth'

model_name = "s2anet_fullrez_crop"
images_folder = "warped"
out_folder = "warped_out"

resize = 1.3

model = init_detector(config_file, checkpoint_file, device='cuda:0')

images = os.listdir(images_folder)
os.makedirs(os.path.join(images_folder, model_name), exist_ok=True)

for img in images:
    if os.path.isdir(os.path.join(images_folder, img)):
        continue
    print(f"Processing {img}")
    img_loaded = cv2.imread(os.path.join(images_folder, img))
    img_loaded = cv2.resize(img_loaded, (int(img_loaded.shape[1]*resize), int(img_loaded.shape[0]*resize)), interpolation = cv2.INTER_AREA)
    result = inference_detector(model, img_loaded)

    det_boxes = rotated_box_to_poly_np(result[1][0][0].cpu().numpy())
    det_labels = result[1][0][1].cpu().numpy()

    img_det = imshow_det_bboxes(img_loaded, det_boxes,
                                det_labels.astype(int) + 1,
                                class_names=CLASSES, show=False, show_label=True, rotated=True)

    cv2.imwrite(os.path.join(out_folder, img), img_det)