from pathlib import Path
from mmdet.apis import init_detector, inference_detector
import os
import cv2
from mmdet.core import rotated_box_to_poly_np
from mmcv.visualization import imshow_det_bboxes
from obb_anns import OBBAnns

annotations_file = "data/deep_scores_dense/deepscores_test.json"
obb = OBBAnns(annotations_file)
obb.load_annotations()
obb.set_annotation_set_filter(['deepscores'])
CLASSES = tuple([v["name"] for (k, v) in obb.get_cats().items()])

config_file = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py'
checkpoint_file = 'checkpoint.pth'

model_name = "s2anet_fullrez_crop"
images_folder = Path("input")
out_folder = Path("input_out")
images_folder.mkdir(exist_ok=True)
out_folder.mkdir(exist_ok=True)

model = init_detector(config_file, checkpoint_file, device='cuda:0')

for file in images_folder.glob('*'):
    if not file.is_file():
        continue
    print(f"Processing {file.name}")
    img_loaded = cv2.imread(str(file))
    # img_loaded = cv2.resize(img_loaded, (int(img_loaded.shape[1]*resize), int(img_loaded.shape[0]*resize)), interpolation = cv2.INTER_AREA)
    result = inference_detector(model, img_loaded)

    det_boxes = rotated_box_to_poly_np(result[1][0][0].cpu().numpy())
    det_labels = result[1][0][1].cpu().numpy()

    img_det = imshow_det_bboxes(img_loaded, det_boxes,
                                det_labels.astype(int) + 1,
                                class_names=CLASSES, show=False, show_label=True, rotated=True)
    cv2.imwrite(out_folder/ file.name, img_det)
print('Done')
