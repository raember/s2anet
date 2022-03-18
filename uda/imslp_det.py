from mmdet.apis import init_detector, inference_detector
import os
import cv2
from mmdet.core import rotated_box_to_poly_np
from mmcv.visualization import imshow_det_bboxes
from obb_anns import OBBAnns
import wandb
import numpy as np

#wandb.init(project="uda", entity="adhirajghosh")
annotations_file = "./data/deep_scores_dense/deepscores_test.json"
obb = OBBAnns(annotations_file)
obb.load_annotations()
obb.set_annotation_set_filter(['deepscores'])
CLASSES = tuple([v["name"] for (k, v) in obb.get_cats().items()])

config_file = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py'
#NOTE: DOING S2anet NOW
checkpoint_file = 'models/deepscoresV2_tugg_halfrez_crop_epoch250.pth'
# checkpoint_file = 'models/epoch_500.pth'

model_name = "s2anet_halfrez_crop"
images_folder = "./uda/test/"
out_folder = "./uda/results/detections/s2anet_half/"
if not os.path.isdir(out_folder):
    print("Created folder ",out_folder)
    os.makedirs(out_folder)

resize = 0.5

model = init_detector(config_file, checkpoint_file, device='cuda:0')

dir = os.listdir(images_folder)
# os.makedirs(os.path.join(images_folder, model_name), exist_ok=True)

for i in dir:
    print(i)
    if not os.path.isdir(os.path.join(out_folder, i)):
        os.makedirs(os.path.join(out_folder, i))
    for img in os.listdir(os.path.join(images_folder, i)):
    # if os.path.isdir(os.path.join(images_folder, img)):
    #     continue
        print(f"Processing {img}")
        img_loaded = cv2.imread(os.path.join(images_folder, i, img))
        if img_loaded is None:
            continue
        img_loaded = cv2.resize(img_loaded, (int(img_loaded.shape[1]*resize), int(img_loaded.shape[0]*resize)), interpolation = cv2.INTER_AREA)
        result = inference_detector(model, img_loaded)
        # print(result[1][0][0].ndim)
        try:
            det_boxes = rotated_box_to_poly_np(result[1][0][0].cpu().numpy())
        except np.AxisError:
            print("No detections")
            continue
        else:
            det_labels = result[1][0][1].cpu().numpy()

            img_det = imshow_det_bboxes(img_loaded, det_boxes,
                                        det_labels.astype(int) + 1,
                                        class_names=CLASSES, show=False, show_label=True, rotated=True)

            cv2.imwrite(os.path.join(out_folder, i, img), img_det)