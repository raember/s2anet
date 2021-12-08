from datetime import datetime
import colorcet as cc
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from obb_anns import OBBAnns
import os.path as osp


def draw_bbox(self, draw, ann, color, oriented, annotation_set=None,
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

    parsed_comments = OBBAnns.parse_comments(ann['comments'])

    if oriented:
        bbox = ann['o_bbox']
        draw.line(bbox + bbox[:2], fill=color, width=3
                  )
    else:
        bbox = ann['a_bbox']
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

ann = OBBAnns('../scanned_ds/ili_scores.json')
ann.load_annotations()
data_root = '../scanned_ds'
out_dir = 'out_ili'
annotation_set = 'deepscores'
annotation_set = ann.annotation_sets.index(annotation_set)
ann.chosen_ann_set = ann.chosen_ann_set[annotation_set]
for img_info in ann.img_info:
    img_id = img_info['id']
    ann.visualize(img_id=img_id, data_root=data_root, out_dir=out_dir)
    # img_info, ann_info = ann.get_img_ann_pair(ids=[img_id])
    # img_info, ann_info = img_info[0], ann_info[0]
    #
    # # Get the data_root from the ann_file path if it doesn't exist
    # if data_root is None:
    #     data_root = osp.split(ann.ann_file)[0]
    #
    # img_dir = osp.join(data_root, 'images')
    #
    # # Get the actual image filepath and the segmentation filepath
    # img_fp = osp.join(img_dir, img_info['filename'])
    # print(f'Visualizing {img_fp}...')
    #
    # # Remember: PIL Images are in form (h, w, 3)
    # img = Image.open(img_fp)
    #
    # # overlay = Image.open(f"../scanned_deepscore_images_png/images/{img_info['filename']}")
    # #
    # # # Here we overlay the segmentation on the original image using the
    # # # colorcet colors
    # # # First we need to get the new color values from colorcet
    # # colors = [ImageColor.getrgb(i) for i in cc.glasbey]
    # # colors = np.array(colors).reshape(768, ).tolist()
    # # colors[0:3] = [0, 0, 0]  # Set background to black
    # #
    # # # Then put the palette
    # # # overlay.putpalette(colors)
    # # overlay_array = np.array(overlay)
    # #
    # # # Now the img and the segmentation can be composed together. Black
    # # # areas in the segmentation (i.e. background) are ignored
    # #
    # # mask = np.zeros_like(overlay_array)
    # # mask[np.where(overlay_array == 0)] = 255
    # # mask = Image.fromarray(mask, mode='L')
    # #
    # # img = Image.composite(img, overlay.convert('RGB'), mask)
    # draw = ImageDraw.Draw(img)
    #
    # # Now draw the gt bounding boxes onto the image
    # for annotation in ann_info.to_dict('records'):
    #     draw = draw_bbox(ann, draw, annotation, '#ed0707', True, annotation_set, False)
    #
    # if ann.proposals is not None:
    #     prop_info = ann.get_img_props(ids=[img_id])
    #
    #     for prop in prop_info.to_dict('records'):
    #         prop_oriented = len(prop['bbox']) == 8
    #         draw = draw_bbox(ann, draw, prop, '#ff0000', prop_oriented)
    #
    # if out_dir is not None:
    #     img.save(osp.join(out_dir, img_info['filename']))
