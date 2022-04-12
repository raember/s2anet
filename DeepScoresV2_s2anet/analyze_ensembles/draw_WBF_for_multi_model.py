# Code implemented based on obb_anns.py and:
# https://github.com/ZFTurbo/Weighted-Boxes-Fusion/ensemble_boxes/ensemble_boxes_wbf.py, 4.1.2022
import argparse
import os
import os.path as osp
import pickle
from itertools import compress, chain
from pathlib import Path
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colors

from DeepScoresV2_s2anet.analyze_ensembles.rotated_ensemble_boxes_wbf import *
from mmdet.datasets import build_dataset


# Functions _draw_bbox_BE and visualize_BE are based on code from obb_anns/obb_anns.py
# https://github.com/raember/obb_anns, 26.1.2022
# Module rotated_ensemble_boxes_wbf is based on code from:
# obb_anns/obb_anns/polyiou, https://github.com/raember/obb_anns, 26.1.2022
# and:
# ensemble-boxes module, https://github.com/ZFTurbo/Weighted-Boxes-Fusion, 26.1.2022


# TODO: Ledger Line and Stem: Make bboxes bigger before wbf and afterwards smaller

def parse_args():
    parser = argparse.ArgumentParser(description='Weighted Box Fusion')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--inp',
        type=str,
        default="work_dirs/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop",
        help="Path to the folder to evaluate")
    parser.add_argument(
        '--out',
        type=str,
        default="work_dirs/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop/analyze_BE_output/",
        help="Path to the output folder")
    parser.add_argument(
        '--s_cache',
        type=str,
        default=None,
        help="Store results in a pickle file (for debugging only)")
    parser.add_argument(
        '--l_cache',
        type=str,
        default=None,
        help="Load the results from a pickle file and only visualize them")
    args = parser.parse_args()
    return args


def _draw_bbox_ensemble(obb, draw, ann, color, oriented, annotation_set=None,
                        print_label=True, print_staff_pos=False, print_onset=False,
                        instances=False, print_score=True, print_score_threshold=0.5, m=1):
    """Draws the bounding box onto an image with a given color.

    :param ImageDraw.ImageDraw draw: ImageDraw object to draw with.
    :param dict ann: Annotation information dictionary of the current
        bounding box to draw.
    :param str color: Color to draw the bounding box in as a hex string,
        e.g. '#00ff00'
    :param bool oriented: Choose between drawing oriented or aligned
        bounding box.
    :param Optional[int] annotation_set: Index of the annotation set to be
        drawn. If None is given, the first one available will be drawn.
    :param Optional[bool] print_label: Determines if the class labels
    are printed on the visualization
    :param Optional[bool] print_score: Determines if the scores
    are printed on the visualization
    :param Optional[bool] print_staff_pos: Determines if the staff positions
    are printed on the visualization
    :param Optional[bool] print_onset:  Determines if the onsets are
    printed on the visualization
    :param Optional[float] print_score_threshold: Only print text, score, etc. if score is below this threshold
    :param Optional[int] m: Number of ensemble members

    :return: The drawn object.
    :rtype: ImageDraw.ImageDraw

    """
    annotation_set = 0 if annotation_set is None else annotation_set
    cat_id = ann['cat_id']
    if isinstance(cat_id, list):
        cat_id = int(cat_id[annotation_set])

    if 'comments' in ann.keys():
        parsed_comments = obb.parse_comments(ann['comments'])

    score_thr = 0.5  # fused score thr: below this value, transparent polygons are plotted.
    if oriented:
        bbox = ann.get('o_bbox', list(ann.get('bbox', [])))
        color = cm.RdYlGn(ann['score'])
        color = colors.rgb2hex(color)
        if ann['score'] < score_thr:
            # color2 = colors.to_rgba(color, alpha=round(1/m, 2))
            # color2 = colors.to_hex(color2, keep_alpha=True)
            draw.polygon(bbox + bbox[:2], outline=color, fill='#ff000040')
        else:
            draw.line(bbox + bbox[:2], fill=color, width=3)
    else:
        bbox = ann.get('a_bbox', list(ann.get('bbox', [])))
        color = cm.RdYlGn(ann['score'])
        color = colors.rgb2hex(color)
        if ann['score'] < score_thr:
            color2 = colors.to_rgba(color, alpha=round(1 / m, 2))
            color2 = colors.to_hex(color2, keep_alpha=True)
            draw.rectangle(bbox, outline=color, width=2, fill='#ff000040')
        else:
            draw.rectangle(bbox, outline=color, width=2)

    # Now draw the label below the bbox
    x0 = min(bbox[::2])
    y0 = max(bbox[1::2])
    pos = (x0, y0)

    def print_text_label(position, text, color_text, color_box):
        x1, y1 = ImageFont.load_default().getsize(text)
        x1 += position[0] + 4
        y1 += position[1] + 4
        draw.rectangle((position[0], position[1], x1, y1), fill=color_box)
        draw.text((position[0] + 2, position[1] + 2), text, color_text)
        return x1, position[1]

    def print_scores(position, text, color_text, score_thr, color_box):
        if float(text) < score_thr:
            x1, y1 = ImageFont.load_default().getsize(text)
            x1 += position[0] + 4
            y1 += position[1] + 4
            draw.rectangle((position[0], position[1], x1, y1), fill=color_box)
            draw.text((position[0] + 2, position[1] + 2), text, color_text)
        else:
            x1, y1 = ImageFont.load_default().getsize(text)
            x1 += position[0] + 4
            y1 += position[1] + 4
            draw.rectangle((position[0], position[1], x1, y1), fill=color_box)
            draw.text((position[0] + 2, position[1] + 2), text, color_text)
        return x1, position[1]

    if instances:
        label = str(int(parsed_comments['instance'].lstrip('#'), 16))
        print_text_label(pos, label, '#ffffff', '#303030')

    else:
        label = obb.cat_info[cat_id]['name']
        score = str(round(ann['score'], 2))
        if label != "stem" and label != "ledgerLine" and ann['score'] < print_score_threshold or (
                label == "stem" or label == "ledgerLine") and ann['score'] < 0.4:
            if print_label:
                pos = print_text_label(pos, label, '#ffffff', color)
            if print_score:
                pos = print_scores(pos, score, '#ffffff', score_thr, color)
            if print_onset and 'onset' in parsed_comments.keys():
                pos = print_text_label(pos, parsed_comments['onset'], '#ffffff',
                                       '#091e94')
            if print_staff_pos and 'rel_position' in parsed_comments.keys():
                print_text_label(pos, parsed_comments['rel_position'],
                                 '#ffffff', '#0a7313')

    return draw


def visualize_ensemble(obb,
                       img_idx=None,
                       img_id=None,
                       data_root=None,
                       out_dir=None,
                       annotation_set=None,
                       oriented=True,
                       instances=False,
                       m=1,
                       show=True):
    """Uses PIL to visualize the ground truth labels of a given image.

    img_idx and img_id are mutually exclusive. Only one can be used at a
    time. If proposals are currently loaded, then also visualizes the
    proposals.

    :param int m: Number of ensemble members
    :param int img_idx: The index of the desired image.
    :param int img_id: The id of the desired image.
    :param Optional[str] data_root: Path to the root data directory. If
        none is given, it is assumed to be the parent directory of the
        ann_file path.
    :param Optional[str] out_dir: Directory to save the visualizations in.
        If a directory is given, then the visualizations produced will also
        be saved.
    :param Optional[str] annotation_set: The annotation set to be
        visualized. If None is given, then the first annotation set
        available will be visualized.
    :param Optional[bool] oriented: Whether to show aligned or oriented
        bounding boxes. A value of True means it will show oriented boxes.
    :param bool show: Whether or not to use pillow's show() method to
        visualize the image.
    :param bool instances: Choose whether to show classes or instances. If
        False, then shows classes. Else, shows instances as the labels on
        bounding boxes.
    """
    # Since we can only visualize a single image at a time, we do i[0] so
    # that we don't have to deal with lists. get_img_ann_pair() returns a
    # tuple that's why we use list comprehension
    img_idx = [img_idx] if img_idx is not None else None
    img_id = [img_id] if img_id is not None else None

    if annotation_set is None:
        annotation_set = 0
        obb.chosen_ann_set = obb.annotation_sets[0]
    else:
        annotation_set = obb.annotation_sets.index(annotation_set)
        obb.chosen_ann_set = obb.chosen_ann_set[annotation_set]

    img_info, ann_info = [i[0] for i in
                          obb.get_img_ann_pair(
                              idxs=img_idx, ids=img_id)]

    # Get the data_root from the ann_file path if it doesn't exist
    if data_root is None:
        data_root = osp.split(obb.ann_file)[0]

    img_dir = osp.join(data_root, 'images')
    seg_dir = osp.join(data_root, 'segmentation')
    inst_dir = osp.join(data_root, 'instance')

    # Get the actual image filepath and the segmentation filepath
    img_fp = osp.join(img_dir, img_info['filename'])
    print(f'Visualizing {img_fp}...')

    # Remember: PIL Images are in form (h, w, 3)
    img = Image.open(img_fp)
    draw = ImageDraw.Draw(img, 'RGBA')

    # Draw the proposals
    if obb.proposals is not None:
        prop_info = obb.get_img_props(idxs=img_idx, ids=img_id)

        for prop in prop_info.to_dict('records'):
            prop_oriented = len(prop['bbox']) == 8
            # Use alpha = 1/m; m = size of ensemble. If all props overlap alpha = 1.
            draw = _draw_bbox_ensemble(obb, draw, prop, '#ff436408', prop_oriented, m)

    if show:
        plt.figure(figsize=(25, 36))
        plt.imshow(img)
        plt.tight_layout()
        plt.show()
        # img.show()
    if out_dir is not None:
        img.save(osp.join(out_dir, 'props_' + img_info['filename']))


def get_model_names(args):
    return sorted(
        [x.split('_', 1)[-1] for x in os.listdir(args.inp) if "result_" in x and not "metrics.csv" in x])


class BboxHelper:

    def __init__(self, bbox):
        self.bbox = bbox
        self.bbox_sorted = self.sort_bboxes()

    def sort_bboxes(self):

        def algo(x):
            """ Sort points in CCW order """
            return (math.atan2(x[0] - np.mean(x_coords), x[1] - np.mean(y_coords)) + 2 * np.pi) % (2 * np.pi)

        y_coords = [self.bbox[1], self.bbox[3], self.bbox[5], self.bbox[7]]
        x_coords = [self.bbox[0], self.bbox[2], self.bbox[4], self.bbox[6]]

        # Sort points ccw
        sorted_points = [(self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (self.bbox[4], self.bbox[5]), (self.bbox[6], self.bbox[7])]
        sorted_points.sort(key=algo)

        # define P1 as lower left -> assume, it is closest to coord 0,0
        dist_to_0 = [np.sqrt(sorted_points[i][0]**2 + sorted_points[i][1]**2) for i in range(4)]
        p1_pos = np.argmin(dist_to_0)
        new_ordering = [0, 3, 2, 1]
        new_ordering = new_ordering[-p1_pos:] + new_ordering[:-p1_pos]

        result = [sorted_points[i] for i in new_ordering]

        # fig, ax = plt.subplots()
        # label = ['p1', 'p2', 'p3', 'p4']
        # x = [result[i][0] for i in range(4)]
        # y = [result[i][1] for i in range(4)]
        # plt.scatter(x=x, y=y)
        # for i, txt in enumerate(label):
        #     ax.annotate(txt, (x[i], y[i]))
        # plt.show()

        return np.array(list(chain.from_iterable(result)))

    def get_sorted_angle_zero(self, add_x=0., add_y=0., width=None, height=None):
        x1, x2 = np.mean([self.bbox_sorted[0], self.bbox_sorted[6]]), np.mean(
            [self.bbox_sorted[2], self.bbox_sorted[4]])
        y1, y2 = np.mean([self.bbox_sorted[1], self.bbox_sorted[3]]), np.mean(
            [self.bbox_sorted[5], self.bbox_sorted[7]])

        if width is not None:
            x = np.mean([x1, x2])
            x1, x2 = x-width/2, x+width/2

        if height is not None:
            y = np.mean([y1, y2])
            y1, y2 = y - height / 2, y + height / 2

        x1 -= add_x / 2
        x2 += add_x / 2
        y1 -= add_y / 2
        y2 += add_y / 2

        fig, ax = plt.subplots()
        label = ['p1', 'p2', 'p3', 'p4']
        x = [self.bbox_sorted[i*2] for i in range(4)]
        y = [self.bbox_sorted[i*2+1] for i in range(4)]
        plt.scatter(x=x, y=y)
        for i, txt in enumerate(label):
            ax.annotate(txt, (x[i], y[i]))

        x = [x1, x2, x2, x1]
        y = [y1, y1, y2, y2]
        plt.scatter(x=x, y=y)
        for i, txt in enumerate(label):
            ax.annotate(txt, (x[i], y[i]))

        plt.show()

        return np.array([x1, y1, x2, y1, x2, y2, x1, y2])


def load_proposals(args, dataset, models):
    boxes_list = []
    scores_list = []
    labels_list = []
    img_idx_list = []

    for i in models:
        json_result_fp = osp.join(args.inp, f"result_{i}/deepscores_results.json")
        dataset.obb.load_proposals(json_result_fp)

        #### WBF Preprocessing
        props = dataset.obb.proposals
        for i, row in props.iterrows():
            if row.cat_id == 42 or row.cat_id == 2:
                if row.cat_id == 2:
                    # make ledger line
                    # y1 -= 0.2
                    # y2 += 0.2
                    props.at[i, 'bbox'] = BboxHelper(row.bbox).get_sorted_angle_zero(add_y=15)
                if row.cat_id == 42:
                    # make stem bigger
                    # x1 -= 0.2
                    # x2 += 0.2
                    props.at[i, 'bbox'] = BboxHelper(row.bbox).get_sorted_angle_zero(add_x=15)

        boxes_list.append(props['bbox'].to_list())
        scores_list.append(list(map(float, props['score'].to_list())))
        labels_list.append(props['cat_id'].to_list())
        img_idx_list.append(props['img_idx'])
        print(f"Adding proposals from ensemble member {i}.")
    # Calculate proposals_WBF
    max_img_idx = max([max(i) for i in img_idx_list])
    # TODO: use different threshold for ledger line and stem (e.g. 0.01)
    iou_thr = 0.1  # This is the most important hyper parameter; IOU of proposal with fused box.
    skip_box_thr = 0.00001  # Skips proposals if score < thr; However, nms is applied when using routine in test_BE.py and score_thr from config applies already.
    # score_thr: value is set below; skips visualization if fused score is below score_thr
    weights = None  # Could weight proposals from a specific ensemble member
    proposals_WBF = []
    # rotated_weighted_boxes mixes boxes from different images during calculation.
    # Thus, execute it on proposals of each image seperately, then concatenate results.
    for i in range(max_img_idx + 1):
        # s: select_proposals_by_img_idx
        s = [img_idx == i for img_idx in img_idx_list]
        boxes_list_i = [list(compress(boxes_list[j], s[j])) for j in
                        range(len(boxes_list))]
        scores_list_i = [list(compress(scores_list[j], s[j])) for j in
                         range(len(scores_list))]
        labels_list_i = [list(compress(labels_list[j], s[j])) for j in
                         range(len(labels_list))]
        img_idx_list_i = [list(compress(img_idx_list[j], s[j])) for j in
                          range(len(img_idx_list))]

        boxes, scores, labels = rotated_weighted_boxes_fusion(boxes_list_i,
                                                              scores_list_i,
                                                              labels_list_i,
                                                              weights=weights,
                                                              iou_thr=iou_thr,
                                                              skip_box_thr=skip_box_thr)

        boxes = pd.Series(list(boxes))
        labels = pd.Series(list(labels)).astype(int)
        img_idxs = pd.Series(chain(*img_idx_list_i))
        scores = pd.Series(list(scores))
        zipped = list(zip(boxes, labels, img_idxs, scores))

        proposals_WBF_i = pd.DataFrame(zipped, columns=['bbox', 'cat_id', 'img_idx',
                                                        'score'])
        proposals_WBF.append(proposals_WBF_i)

    proposals_WBF = pd.concat(proposals_WBF)
    return proposals_WBF


def store_proposals(args, proposals_WBF):
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    out_file = os.path.join(args.out, 'proposals_WBF.pkl')
    pickle.dump(proposals_WBF, open(out_file, 'wb'))


def evaluate_wbf_performance(args, cfg, proposals_WBF):
    proposals_WBF_per_img = []
    for img_idx in sorted(set(list(proposals_WBF.img_idx))):
        props = proposals_WBF[proposals_WBF.img_idx == img_idx]
        result_prop = [np.empty((0, 9))] * 136

        for cat_id in sorted(set(list(props.cat_id))):
            props_cat = props[props.cat_id == cat_id]
            result_prop[cat_id - 1] = np.concatenate((np.array(list(props_cat.bbox)),
                                                      np.array(list(props_cat.score)).reshape(
                                                          (np.array(list(props_cat.score)).shape[0], 1))), axis=1)

        proposals_WBF_per_img.append(result_prop)
    dataset = build_dataset(cfg.data.test)
    metrics = dataset.evaluate(proposals_WBF_per_img,
                               result_json_filename=str(Path(args.out) / "deepscores_ensemble_results.json"),
                               work_dir=args.out)
    print("Mean AP for Threshold=0.5 is ", np.mean([v[0.5]['ap'] for v in metrics.values()]))
    out_file = open(os.path.join(args.out, "deepscores_ensemble_metrics.pkl"), 'wb')
    pickle.dump(metrics, out_file)
    out_file.close()
    return dataset


def postprocess_proposals(proposals_WBF):
    # make all stem vertical and ledger line horizontal
    proposals_WBF = proposals_WBF.reset_index(drop=True)
    for i, row in proposals_WBF.iterrows():
        if row.cat_id == 42 or row.cat_id == 2:
            if row.cat_id == 2:
                # make ledger line
                proposals_WBF.at[i, 'bbox'] = BboxHelper(row.bbox).get_sorted_angle_zero(height=3)
            if row.cat_id == 42:
                # make stem bigger
                proposals_WBF.at[i, 'bbox'] = BboxHelper(row.bbox).get_sorted_angle_zero(width=2)
    return proposals_WBF

def visualize_proposals(args, dataset, m, proposals_WBF):
    # Create output directory
    out_dir = osp.join(args.out, "visualized_proposals/")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    # Dropping proposals with an average score < score_thr (e.g. if 1 member makes a proposal with score 0.3 and all others make no proposal; the fused score is: 0.3/30=0.01)
    score_thr = 0.3  # TODO: DELME (IMPLEMENTED AGAIN LATER ON)
    proposals_WBF = proposals_WBF.drop(proposals_WBF[proposals_WBF.score < score_thr].index)
    # Drop 'staff'-class (looks ugly on plot)
    # proposals_WBF = proposals_WBF.drop(proposals_WBF[proposals_WBF.cat_id == 135].index)
    dataset.obb.proposals = proposals_WBF
    for img_info in dataset.obb.img_info:
        visualize_ensemble(obb=dataset.obb,
                           img_id=img_info['id'],
                           data_root=dataset.data_root,
                           out_dir=out_dir,
                           m=m)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    if not Path(args.l_cache).exists():
        dataset = build_dataset(cfg.data.test)

        # Deduce m (number of BatchEnsemble members)
        models = get_model_names(args)
        models = models[21::3]  # TODO: delme (less models for debugging)
        m = len(models)

        proposals_WBF = load_proposals(args, dataset, models)
        proposals_WBF = postprocess_proposals(proposals_WBF)
        store_proposals(args, proposals_WBF)

        dataset = evaluate_wbf_performance(args, cfg, proposals_WBF)

        if args.s_cache is not None:
            data = {
                'proposals_WBF': proposals_WBF,
                'dataset': dataset,
                'm': m,
            }
            with open(args.s_cache, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(args.l_cache, 'rb') as handle:
            data = pickle.load(handle)
            proposals_WBF = data['proposals_WBF']
            dataset = data['dataset']
            m = data['m']

        visualize_proposals(args, dataset, m, proposals_WBF)





if __name__ == '__main__':
    main()
