import functools
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, send_from_directory, Response, make_response
from mmcv import imshow_det_bboxes

from DeepScoresV2_s2anet.analyze_ensembles.wbf_rotated_boxes import rotated_weighted_boxes_fusion
from mmdet.apis import init_detector, inference_detector
from mmdet.core import outputs_rotated_box_to_poly_np, poly_to_rotated_box_np, rotated_box_to_poly_np

UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

config_path = "configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py"
models_checkp_paths = ["checkpoint.pth"]

#config_path = "s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py"
#models_checkp_paths = ["aug_epoch_2000.pth"]

class_names = (
    'brace', 'ledgerLine', 'repeatDot', 'segno', 'coda', 'clefG', 'clefCAlto', 'clefCTenor', 'clefF',
    'clefUnpitchedPercussion', 'clef8', 'clef15', 'timeSig0', 'timeSig1', 'timeSig2', 'timeSig3', 'timeSig4',
    'timeSig5', 'timeSig6', 'timeSig7', 'timeSig8', 'timeSig9', 'timeSigCommon', 'timeSigCutCommon',
    'noteheadBlackOnLine', 'noteheadBlackOnLineSmall', 'noteheadBlackInSpace', 'noteheadBlackInSpaceSmall',
    'noteheadHalfOnLine', 'noteheadHalfOnLineSmall', 'noteheadHalfInSpace', 'noteheadHalfInSpaceSmall',
    'noteheadWholeOnLine', 'noteheadWholeOnLineSmall', 'noteheadWholeInSpace', 'noteheadWholeInSpaceSmall',
    'noteheadDoubleWholeOnLine', 'noteheadDoubleWholeOnLineSmall', 'noteheadDoubleWholeInSpace',
    'noteheadDoubleWholeInSpaceSmall', 'augmentationDot', 'stem', 'tremolo1', 'tremolo2', 'tremolo3', 'tremolo4',
    'tremolo5', 'flag8thUp', 'flag8thUpSmall', 'flag16thUp', 'flag32ndUp', 'flag64thUp', 'flag128thUp', 'flag8thDown',
    'flag8thDownSmall', 'flag16thDown', 'flag32ndDown', 'flag64thDown', 'flag128thDown', 'accidentalFlat',
    'accidentalFlatSmall', 'accidentalNatural', 'accidentalNaturalSmall', 'accidentalSharp', 'accidentalSharpSmall',
    'accidentalDoubleSharp', 'accidentalDoubleFlat', 'keyFlat', 'keyNatural', 'keySharp', 'articAccentAbove',
    'articAccentBelow', 'articStaccatoAbove', 'articStaccatoBelow', 'articTenutoAbove', 'articTenutoBelow',
    'articStaccatissimoAbove', 'articStaccatissimoBelow', 'articMarcatoAbove', 'articMarcatoBelow', 'fermataAbove',
    'fermataBelow', 'caesura', 'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th',
    'rest32nd', 'rest64th', 'rest128th', 'restHNr', 'dynamicP', 'dynamicM', 'dynamicF', 'dynamicS', 'dynamicZ',
    'dynamicR', 'graceNoteAcciaccaturaStemUp', 'graceNoteAppoggiaturaStemUp', 'graceNoteAcciaccaturaStemDown',
    'graceNoteAppoggiaturaStemDown', 'ornamentTrill', 'ornamentTurn', 'ornamentTurnInverted', 'ornamentMordent',
    'stringsDownBow', 'stringsUpBow', 'arpeggiato', 'keyboardPedalPed', 'keyboardPedalUp', 'tuplet3', 'tuplet6',
    'fingering0', 'fingering1', 'fingering2', 'fingering3', 'fingering4', 'fingering5', 'slur', 'beam', 'tie',
    'restHBar', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin', 'tuplet1', 'tuplet2', 'tuplet4', 'tuplet5',
    'tuplet7', 'tuplet8', 'tuplet9', 'tupletBracket', 'staff', 'ottavaBracket'
)

app = Flask(__name__)


# TODO: if enough memory -> make maxsize = len(pretrained_models)
@functools.lru_cache(maxsize=1)
def _get_model(checkpoint_pth):
    return init_detector(config_path, checkpoint_pth, device='cuda:0')


def _get_detections_from_pred(predictions):
    detect_list = []
    bboxes, classes = predictions[0][1][0]

    for bbox, class_id in zip(bboxes, classes):
        out = list(map(int, map(torch.round, bbox)))
        out[4] = float(torch.round(bbox[4] * 10000)) / 10000  # Torch version 1.8.0 does not support decimals
        out[5] = class_names[int(class_id) + 1]
        detect_list.append(out)

    return detect_list


def _get_detections_from_pred_multimodel(predictions):
    boxes, scores, labels = [], [], []

    predictions = [outputs_rotated_box_to_poly_np([x[0]])[0] for x in predictions]
    for prediction in predictions:
        boxes_i, scores_i, labels_i = [], [], []
        for c, class_pred in enumerate(prediction):
            if class_pred.shape[0] > 0:
                boxes_i += list(class_pred[:, :-1])
                scores_i += list(class_pred[:, -1])
                labels_i += [c] * class_pred.shape[0]
        boxes.append(boxes_i)
        scores.append(scores_i)
        labels.append(labels_i)

    boxes, scores, labels = rotated_weighted_boxes_fusion(boxes, scores, labels,
                                                          weights=None, iou_thr=0.3, skip_box_thr=0.00001)
    boxes = poly_to_rotated_box_np(boxes)

    # TODO: add postprocessing stem & ledgerLine ? -> currently under development

    detect_list = []
    labels_str = [class_names[int(id_) + 1] for id_ in labels.tolist()]
    for b, s, c in zip(boxes.tolist(), scores.tolist(), labels_str):
        out = list(map(round, b))
        out[4] = round(b[4], 4)
        out += [round(s, 4)] + [c]
        detect_list.append(out)

    return detect_list


@app.route('/')
def hello_world():
    return 'Welcome to the classifier, go to /classify to start classifying'


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        file = request.files['image_patch']
        if file and allowed_file(file.filename):

            pic = Image.open(file).convert('RGB')
            img = np.asarray(pic)

            predictions = []
            for checkpoint_pth in models_checkp_paths:
                model = _get_model(checkpoint_pth)
                predictions.append(inference_detector(model, img))  # returns a tuple: list

            if len(predictions) == 1:
                detect_list = _get_detections_from_pred(predictions)

            else:
                detect_list = _get_detections_from_pred_multimodel(predictions)

            detect_list = bbox_translate(detect_list)
            print(f"Detected {len(detect_list)} bboxes")
            detect_dict = dict(bounding_boxes=detect_list)
            return Response(json.dumps(detect_dict), mimetype='application/json')
        else:
            return Response('Unsupported filetype', mimetype='application/json')
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image_patch>
         <input type=submit value=Upload>
    </form>
    """


@app.route('/classify_img', methods=['GET', 'POST'])
def classify_img():
    if request.method == 'POST':
        file = request.files['image_patch']
        if file and allowed_file(file.filename):

            pic = Image.open(file).convert('RGB')
            img = np.asarray(pic)

            predictions = []
            for checkpoint_pth in models_checkp_paths:
                model = _get_model(checkpoint_pth)
                predictions.append(inference_detector(model, img))  # returns a tuple: list

            det_boxes = predictions[0][1][0][0].cpu().numpy()
            det_labels = predictions[0][1][0][1].cpu().numpy()
            det_boxes = bbox_translate(det_boxes)
            det_boxes = rotated_box_to_poly_np(det_boxes)
            img_det = imshow_det_bboxes(img, det_boxes,
                                        det_labels.astype(int) + 1,
                                        class_names=list(class_names), show=False, show_label=True, rotated=True)
            ann_img = Image.fromarray(img_det, "RGB")

            img_io = BytesIO()
            ann_img.save(img_io, 'png')
            img_io.seek(0)
            return Response(img_io, mimetype='image/png')
        else:
            return Response('Unsupported filetype', mimetype='application/json')
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=image_patch>
         <input type=submit value=Upload>
    </form>
    """

# enforce w to be width and h to be height
def bbox_translate(det_boxes):
    for det in det_boxes:
        if det[4] > np.pi/4:
            det[4] = det[4] - np.pi/2
            store = det[2]
            det[2] = det[3]
            det[3] = store

    return det_boxes


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0')
