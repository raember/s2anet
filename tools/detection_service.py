import functools
import json
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, request, send_from_directory, Response
from mmcv import imshow_det_bboxes

from DeepScoresV2_s2anet.analyze_ensembles.wbf_rotated_boxes import rotated_weighted_boxes_fusion
from DeepScoresV2_s2anet.omr_prototype_alignment import prototype_alignment, render
from mmdet.apis import init_detector, inference_detector
from mmdet.core import outputs_rotated_box_to_poly_np, poly_to_rotated_box_np, rotated_box_to_poly_np

UPLOAD_FOLDER = './Patches/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
ID_TO_CLASS = render.Render(0, 0, 0).id_to_class

config_path = "configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py"
models_checkp_paths = ["data/deepscoresV2_tugg_halfrez_crop_epoch250.pth", "data/deepscoresV2_tugg_halfrez_crop_epoch250.pth"]

# config_path = "s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py"
# models_checkp_paths = ["aug_epoch_2000.pth"]

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


# TODO: if enough memory -> make maxsize = len(pretrained_models), else maxsize=1
@functools.lru_cache(maxsize=len(models_checkp_paths))
def _get_model(checkpoint_pth):
    return init_detector(config_path, checkpoint_pth, device='cuda:0')


def _postprocess_bboxes(img, boxes, labels):
    img = Image.fromarray(img)
    proposal_list = [{'proposal': np.append(box[:5], class_names[int(label) + 1])} for box, label in zip(boxes, labels)]
    processed_proposals = prototype_alignment._process_single(img, proposal_list,
                                                              whitelist=["key", "clef", "accidental", "notehead"])
    new_boxes = np.zeros(boxes.shape)
    new_boxes[..., :5] = np.stack(processed_proposals)
    if new_boxes.shape[1] == 6:
        # copy scores
        new_boxes[..., 5] = boxes[..., 5]
    return new_boxes


# enforce w to be width and h to be height
def bbox_translate(det_boxes):
    for det in det_boxes:
        if det[4] > np.pi / 4:
            det[4] = det[4] - np.pi / 2
            store = det[2]
            det[2] = det[3]
            det[3] = store

    return det_boxes


def unravel_predictions(predictions):
    boxes = predictions[0][1][0][0].cpu().numpy()
    labels = predictions[0][1][0][1].cpu().numpy()

    return boxes, labels


def _apply_wbf(predictions):
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

    return boxes, scores, labels


def _get_detections_from_pred(bboxes, labels):
    detect_list = []

    for bbox, class_id in zip(bboxes, labels):
        out = list(map(int, map(round, bbox)))
        out[4] = round(float(bbox[4]), 4)
        out[5] = class_names[int(class_id) + 1]
        detect_list.append(out)

    return detect_list


def _get_detections_from_pred_multimodel(boxes, scores, labels):
    detect_list = []
    labels_str = [class_names[int(id_) + 1] for id_ in labels.tolist()]
    for b, s, c in zip(boxes.tolist(), scores.tolist(), labels_str):
        out = list(map(round, b))
        out[4] = round(b[4], 4)
        out += [round(s, 4)] + [c]
        detect_list.append(out)

    return detect_list


def _classification(pred_processing):
    if request.method == 'POST':
        file = request.files['image_patch']
        if file and allowed_file(file.filename):

            pic = Image.open(file).convert('RGB')
            img = np.asarray(pic)

            predictions = []
            for checkpoint_pth in models_checkp_paths:
                model = _get_model(checkpoint_pth)
                prediction = inference_detector(model, img)
                predictions.append(prediction)  # returns a tuple: list

            is_multimodel = len(predictions) != 1

            if is_multimodel:
                boxes, scores, labels = _apply_wbf(predictions)
                boxes = poly_to_rotated_box_np(boxes)
            else:
                boxes, labels = unravel_predictions(predictions)
                scores = None

            print(f"Detected {len(boxes)} bboxes")
            boxes = bbox_translate(boxes)

            # can be called with argument in url, e.g. http://myserver:5000/classify?postprocess=True
            do_postprocessing = request.args.get("postprocess", default=False, type=lambda v: v.lower() == 'true')
            if do_postprocessing:
                boxes = _postprocess_bboxes(img, boxes, labels)

            return pred_processing(boxes, scores, labels, file, is_multimodel)

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


@app.route('/')
def hello_world():
    return 'Welcome to the classifier, go to /classify to start classifying'


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    def pred_processing(boxes, scores, labels, file, is_multimodel):
        if is_multimodel:
            detect_list = _get_detections_from_pred_multimodel(boxes, scores, labels)
        else:
            detect_list = _get_detections_from_pred(boxes, labels)

        detect_dict = dict(bounding_boxes=detect_list)

        return Response(json.dumps(detect_dict), mimetype='application/json')

    return _classification(pred_processing)


@app.route('/classify_img', methods=['GET', 'POST'])
def classify_img():
    def pred_processing(boxes, scores, labels, file, is_multimodel):
        boxes = rotated_box_to_poly_np(boxes)

        pic = Image.open(file).convert('RGB')
        img = np.asarray(pic)

        img_det = imshow_det_bboxes(img, boxes, labels.astype(int) + 1,
                                    class_names=list(class_names), show=False, show_label=True, rotated=True)
        ann_img = Image.fromarray(img_det, "RGB")

        img_io = BytesIO()
        ann_img.save(img_io, 'png')
        img_io.seek(0)
        return Response(img_io, mimetype='image/png')

    return _classification(pred_processing)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    # print("Setup cache for post-processing...", end=" ")
    # render.fill_cache()
    # print("Done")
    app.run(host='0.0.0.0')  # 1:42
