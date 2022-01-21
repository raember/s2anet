from mmdet.datasets import build_dataset
import os

import os.path as osp
from PIL import Image, ImageColor, ImageDraw, ImageFont
import numpy as np
import colorcet as cc


def _draw_bbox_BE(self, draw, ann, color, oriented, annotation_set=None,
                  print_label=False, print_staff_pos=False, print_onset=False,
                  instances=False):
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
    :param Optional[bool] print_staff_pos: Determines if the staff positions
    are printed on the visualization
    :param Optional[bool] print_onset:  Determines if the onsets are
    printed on the visualization

    :return: The drawn object.
    :rtype: ImageDraw.ImageDraw
    """
    annotation_set = 0 if annotation_set is None else annotation_set
    cat_id = ann['cat_id']
    if isinstance(cat_id, list):
        cat_id = int(cat_id[annotation_set])
    
    if 'comments' in ann.keys():
        parsed_comments = self.parse_comments(ann['comments'])
    
    if oriented:
        bbox = ann.get('o_bbox', list(ann.get('bbox', [])))
        draw.polygon(bbox + bbox[:2], fill=color)
    else:
        bbox = ann['a_bbox']
        draw.rectangle(bbox, fill=color)
    
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
    
    if instances:
        label = str(int(parsed_comments['instance'].lstrip('#'), 16))
        print_text_label(pos, label, '#ffffff', '#303030')
    
    else:
        label = self.cat_info[cat_id]['name']
        
        if print_label:
            pos = print_text_label(pos, label, '#ffffff', '#303030')
        if print_onset and 'onset' in parsed_comments.keys():
            pos = print_text_label(pos, parsed_comments['onset'], '#ffffff',
                                   '#091e94')
        if print_staff_pos and 'rel_position' in parsed_comments.keys():
            print_text_label(pos, parsed_comments['rel_position'],
                             '#ffffff', '#0a7313')
    
    return draw


def visualize_BE(self,
                 img_idx=None,
                 img_id=None,
                 data_root=None,
                 out_dir=None,
                 annotation_set=None,
                 oriented=True,
                 instances=False,
                 show=True,
                 member=0):
    """Uses PIL to visualize the ground truth labels of a given image.

    img_idx and img_id are mutually exclusive. Only one can be used at a
    time. If proposals are currently loaded, then also visualizes the
    proposals.

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
    :param int member: Index of current ensemble member (i in loop)
    """
    # Since we can only visualize a single image at a time, we do i[0] so
    # that we don't have to deal with lists. get_img_ann_pair() returns a
    # tuple that's why we use list comprehension
    img_idx = [img_idx] if img_idx is not None else None
    img_id = [img_id] if img_id is not None else None
    
    if annotation_set is None:
        annotation_set = 0
        self.chosen_ann_set = self.annotation_sets[0]
    else:
        annotation_set = self.annotation_sets.index(annotation_set)
        self.chosen_ann_set = self.chosen_ann_set[annotation_set]
    
    img_info, ann_info = [i[0] for i in
                          self.get_img_ann_pair(
                              idxs=img_idx, ids=img_id)]
    
    # Get the data_root from the ann_file path if it doesn't exist
    if data_root is None:
        data_root = osp.split(self.ann_file)[0]
    
    img_dir = osp.join(data_root, 'images')
    seg_dir = osp.join(data_root, 'segmentation')
    inst_dir = osp.join(data_root, 'instance')
    
    # Get the actual image filepath and the segmentation filepath
    # If proposals have already been visualized for an image load these
    if member == 0:
        img_fp = osp.join(img_dir, img_info['filename'])
        print(f'Visualizing {img_fp}...')
    else:
        prop_dir = '/s2anet/DeepScoresV2_s2anet/analyze_BE_output/visualized_proposals'
        img_fp = osp.join(prop_dir, 'props_' + img_info['filename'])
        if not osp.exists(img_fp):  # If there were no proposals for the current image until now (i.e. none of the other members made any proposals yet)
            img_fp = osp.join(img_dir, img_info['filename'])
        print(f'Visualizing {img_fp}...')
    
    # Remember: PIL Images are in form (h, w, 3)
    img = Image.open(img_fp)
    
    # if instances:
    #     # Do stuff
    #     inst_fp = osp.join(
    #         inst_dir,
    #         osp.splitext(img_info['filename'])[0] + '_inst.png'
    #     )
    #     overlay = Image.open(inst_fp)
    #     img.putalpha(255)
    #     img = Image.alpha_composite(img, overlay)
    #     img = img.convert('RGB')
    #
    # else:
    #     seg_fp = osp.join(
    #         seg_dir,
    #         osp.splitext(img_info['filename'])[0] + '_seg.png'
    #     )
    #     overlay = Image.open(seg_fp)
    #
    #     # Here we overlay the segmentation on the original image using the
    #     # colorcet colors
    #     # First we need to get the new color values from colorcet
    #     colors = [ImageColor.getrgb(i) for i in cc.glasbey]
    #     colors = np.array(colors).reshape(768, ).tolist()
    #     colors[0:3] = [0, 0, 0]  # Set background to black
    #
    #     # Then put the palette
    #     overlay.putpalette(colors)
    #     overlay_array = np.array(overlay)
    #
    #     # Now the img and the segmentation can be composed together. Black
    #     # areas in the segmentation (i.e. background) are ignored
    #
    #     mask = np.zeros_like(overlay_array)
    #     mask[np.where(overlay_array == 0)] = 255
    #     mask = Image.fromarray(mask, mode='L')
    #
    #     img = Image.composite(img, overlay.convert('RGB'), mask)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Now draw the gt bounding boxes onto the image
    # for ann in ann_info.to_dict('records'):
    #     draw = self._draw_bbox(draw, ann, '#43ff64d9', oriented,  # Get rgba hexcode from: https://rgbacolorpicker.com/rgba-to-hex, 22.12.21
    #                            annotation_set, instances)
    
    # Draw the proposals
    if self.proposals is not None:
        prop_info = self.get_img_props(idxs=img_idx, ids=img_id)
        
        for prop in prop_info.to_dict('records'):
            prop_oriented = len(prop['bbox']) == 8
            # Use alpha = 1/m; m = size of ensemble. If all props overlap alpha = 1.
            draw = _draw_bbox_BE(self, draw, prop, '#ff436408', prop_oriented)
    
    if show:
        img.show()
    if out_dir is not None:
        img.save(osp.join(out_dir, 'props_' + img_info['filename'])
                 )


def main():
    # Settings from config file that was used to generate
    # the respective proposals
    dataset_type = 'DeepScoresV2Dataset_BE'
    data_root = '/s2anet/data/deep_scores_dense/'
    img_norm_cfg = dict(
        mean=[240, 240, 240],
        std=[57, 57, 57],
        to_rgb=False)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=1.0,
            flip=False,
            transforms=[
                dict(type='RotatedResize', img_scale=1.0, keep_ratio=True),
                dict(type='RotatedRandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    data_dict = dict(
        type=dataset_type,
        ann_file=data_root + 'deepscores_test_small.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        use_oriented_bboxes=True)
    
    json_results_dir = "/s2anet/work_dirs/s2anet_r50_fpn_1x_deepscoresv2_BE"
    work_dir = "/s2anet/DeepScoresV2_s2anet/analyze_BE_output/"
    dataset = build_dataset(data_dict)
    
    # Deduce m (number of BatchEnsemble members)
    for base_i, folders_i, files_i in os.walk(json_results_dir):
        jsons = [x.split('_')[-1] for x in files_i if
                 "deepscores_results_" in x]
    m = max([int(x.split('.')[0]) for x in jsons]) + 1
    
    # Create output directory
    out_dir = osp.join(work_dir, "visualized_proposals/")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    # Loop through proposals of each ensemble member and visualize them
    # on one page.
    for i in range(m):
        
        json_result_fp = osp.join(json_results_dir,
                                   f"deepscores_results_{i}.json")
        dataset.obb.load_proposals(json_result_fp)

        # Drop 'staff'-class (looks ugly on plot)
        props = dataset.obb.proposals
        props = props.drop(
            props[props.cat_id == 135].index)
        dataset.obb.proposals = props
    
        for img_info in dataset.obb.img_info:
            # TODO: If implementation works visualize_BE could be added as an OBBAnns method
            visualize_BE(self=dataset.obb,
                         img_id=img_info['id'],
                         data_root=dataset.data_root,
                         out_dir=out_dir,
                         member=i)


if __name__ == '__main__':
    main()
