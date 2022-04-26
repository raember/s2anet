import functools
import json

import numpy as np
from PIL import Image
from flask import Flask, request, send_from_directory, Response

from DeepScoresV2_s2anet.analyze_ensembles.wbf_rotated_boxes import rotated_weighted_boxes_fusion
from mmdet.apis import init_detector, inference_detector
from mmdet.core import outputs_rotated_box_to_poly_np, poly_to_rotated_box_np

UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

config_path = "configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py"
models_checkp_paths = ["data/deepscoresV2_tugg_halfrez_crop_epoch250.pth",
                       "data/deepscoresV2_tugg_halfrez_crop_epoch250.pth"]  # ["checkpoint.pth"]

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
        out = list(map(round, bbox))
        out[4] = round(bbox[4], 4)  # angle
        out[5] = class_names[int(class_id)]
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
    labels_str = [class_names[int(id_)] for id_ in labels.tolist()]
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
        print(request.headers)
        print(list(request.files.items()))
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


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0')
