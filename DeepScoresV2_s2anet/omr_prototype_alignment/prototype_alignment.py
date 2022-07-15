import math
import time
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List

import cv2
import cv2 as cv
import numpy as np
from PIL import Image as PImage
from PIL.Image import Image
from PIL.ImageChops import invert
from PIL.ImageDraw import Draw
from PIL.ImageOps import grayscale
from matplotlib import cm
from matplotlib import pyplot as plt
from shapely.affinity import rotate
from shapely.geometry import Polygon
from skimage.morphology import binary_erosion

from DeepScoresV2_s2anet.omr_prototype_alignment.glyph_transform import GlyphGenerator
from DeepScoresV2_s2anet.omr_prototype_alignment.optical_flow import optical_flow_merging
from DeepScoresV2_s2anet.omr_prototype_alignment.render import Render, BASE_PATH
from mmdet.core import rotated_box_to_poly_np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# https://www.geogebra.org/calculator
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + ggbApplet.getValue('α').toPrecision(4)
# or
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + ggbApplet.getValue('w').toFixed() + ", " + ggbApplet.getValue('h').toFixed() + ", " + -ggbApplet.getValue('β').toPrecision(4)
# [x_ctr,y_ctr,w,h,angle]

SAMPLES = [
    (str(BASE_PATH / 'images' / 'sample.png'), [
        {
            'proposal': np.array([155, 242, 41, 112, 0.1105, "clefG"]),
            'gt': np.array([156, 240, 43, 116, 0.0, "clefG"])
        }, {
            'proposal': np.array([513, 180, 19, 16, -0.1463, "noteheadBlackOnLine"]),
            'gt': np.array([513, 180, 19, 17, 0.0, "noteheadBlackOnLine"])
        }, {
            'proposal': np.array([513, 163, 71, 14, -0.0841, "slur"]),
            'gt': np.array([512, 164, 75, 18, -0.1401, "slur"])
        }, {
            'proposal': np.array([619, 240, 16, 28, 0.3046, "rest8th"]),
            'gt': np.array([621, 240, 17, 27, 0.0, "rest8th"])
        }, {
            'proposal': np.array([947, 312, 15, 44, 0.6168, "dynamicF"]),
            'gt': np.array([948, 312, 34, 37, 0.0, "dynamicF"])
        }, {
            'proposal': np.array([985, 193, 28, 3, 0.177, "ledgerLine"]),
            'gt': np.array([987, 194, 26, 3, 0.0, "ledgerLine"])
        }, {
            'proposal': np.array([1308, 266, 98, 5, 0.0303, "beam"]),
            'gt': np.array([1310, 266, 100, 7, 0.03434, "beam"])
        }, {
            'proposal': np.array([590, 311, 551, 23, -0.0208, "dynamicCrescendoHairpin"]),
            'gt': np.array([597, 311, 586, 26, 0.0, "dynamicCrescendoHairpin"])
        }, {
            'proposal': np.array([734, 1087, 16, 43, 0.491, "flag8thDown"]),
            'gt': np.array([733, 1088, 19, 44, 0.0, "flag8thDown"])
        },
    ])
]

GLYPH_GEN = GlyphGenerator()


def pad(img: Image, pad: int) -> Image:
    width, height = img.size
    result = PImage.new(img.mode, (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    result.paste(img, (pad, pad))
    return result


def cls_to_glyph(class_name: str, width: int, height: int, angle: float, padding: int) -> Image:
    png_data = Render(
        class_name=class_name, height=height, width=width, csv_path=str(BASE_PATH / 'data' / 'name_uni.csv')).render(
        str(BASE_PATH / 'data' / 'Bravura.svg'), save_svg=False, save_png=False)
    with BytesIO(png_data) as bio:
        img = PImage.open(bio)
        img.load()
        img = img.rotate(angle * 180.0 / math.pi, PImage.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0))
        img = img.transpose(PImage.FLIP_TOP_BOTTOM)
        img = pad(img, padding)
        glyph = PImage.new("RGB", img.size, (255, 255, 255))
        glyph.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        glyph = invert(glyph)
        return grayscale(glyph)


def bbox_translate(bbox: np.ndarray) -> np.ndarray:
    if bbox[4] > np.pi / 4:
        bbox[4] = bbox[4] - np.pi / 2
        bbox[3], bbox[2] = bbox[2], bbox[3]
    return bbox


def get_glyphs(cls: str, bbox: np.ndarray, padding: int = 50) -> List[Image]:
    _, _, w, h, a = bbox
    glyph = cls_to_glyph(cls, w, h, a, padding)
    if cls in ['tie', 'slur']:
        return [glyph, glyph.transpose(PImage.FLIP_TOP_BOTTOM)]
    return [glyph]


def extract_bbox_from(glyph: Image, prop_bbox: np.ndarray, cls: str) -> np.ndarray:
    x1, y1, _, _, a = prop_bbox
    angle = 0.0
    if cls in ['tie', 'slur', 'beam']:  # Transfer angle if tie, slur etc.
        angle = a
    rectified_glyph = glyph.rotate(-angle * 180.0 / math.pi, PImage.BILINEAR, fillcolor=0)
    bbox = rectified_glyph.getbbox()
    if bbox is None:
        return prop_bbox
    bx1, by1, bx2, by2 = bbox
    cx1, cy1 = glyph.size
    cx1 /= 2
    cy1 /= 2
    w = bx2 - bx1
    h = by2 - by1
    cx2 = bx1 + w / 2
    cy2 = by1 + h / 2
    cxdiff = cx2 - cx1
    cydiff = cy2 - cy1
    x2 = x1 + cxdiff
    y2 = y1 + cydiff
    return np.array([int(x2), int(y2), int(w), int(h), angle])


def get_roi(img, bbox):
    area_size = (np.sqrt(bbox[2] ** 2 + bbox[3] ** 2)) / 2 + 20
    x_min, x_max = int(max(0, np.floor(bbox[0] - area_size))), int(min(img.shape[1], np.ceil(bbox[0] + area_size)))
    y_min, y_max = int(max(0, np.floor(bbox[1] - area_size))), int(min(img.shape[0], np.ceil(bbox[1] + area_size)))

    return img[y_min:y_max, x_min:x_max], (y_min, x_max)


def generate_video(imgs):
    import matplotlib.animation as animation

    frames = []
    fig = plt.figure()
    for img in imgs:
        frames.append([plt.imshow(img, cmap=cm.jet, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

    path = Path("process2_debugging")
    if not path.exists():
        path.mkdir()

    ani.save(str(path / f'debug_{datetime.now().strftime("%Y%m%d-%H%M%S")}.mp4'))


def _create_search_list(lower, upper, stepsize=1):
    center = (lower + upper) / 2
    list_1, list_2 = np.arange(lower, center, stepsize), np.arange(center, upper, stepsize)
    return [ab for a, b in zip(list_1[::-1], list_2) for ab in (a, b)]


def process2(img_np, bbox: np.ndarray, glyph: Image, cls: str, store_video: bool = False) -> Image:
    if len(bbox) == 0:
        return

    img_roi, (y_off, x_off) = get_roi(img_np, bbox)

    orig_angle = bbox[4]

    best_glyph, best_overlap = None, -1

    angles = _create_search_list(orig_angle - 0.2, orig_angle + 0.2, 0.05)
    x_shifts = _create_search_list(img_roi.shape[0] // 2 - 5, img_roi.shape[0] // 2 + 5)
    y_shifts = _create_search_list(img_roi.shape[1] // 2 - 5, img_roi.shape[1] // 2 + 5)


    assert len(x_shifts) > 0 and len(y_shifts) > 0

    imgs, i = [], 0

    count_angle_not_improved = 0
    for angle in angles:
        count_xshift_not_improved = 0
        for x_shift in x_shifts:
            count_yshift_not_improved = 0
            for y_shift in y_shifts:

                padding_left = x_shift
                padding_right = img_roi.shape[1] - padding_left
                padding_top = y_shift
                padding_bottom = img_roi.shape[0] - padding_top

                proposed_glyph = GLYPH_GEN.get_transformed_glyph(cls, int(bbox[3]), int(bbox[2]), angle, padding_left,
                                                                 padding_right, padding_top, padding_bottom)

                if store_video and i % 100 == 0:
                    img = img_roi.copy()
                    img = img * 1
                    img[proposed_glyph] = 2
                    imgs.append(img)
                i += 1

                overlap = np.average(img_roi[proposed_glyph])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_glyph = proposed_glyph
                    best_box = np.array([x_off+x_shift, y_off+y_shift, bbox[2], bbox[3], angle])
                    count_angle_not_improved, count_xshift_not_improved, count_yshift_not_improved = 0, 0, 0

                if count_yshift_not_improved > 2:
                    break
            if count_xshift_not_improved > 2:
                break
        if count_angle_not_improved > 3:
            break

    if store_video:
        img = img_roi.copy()
        img = img * 1
        img[best_glyph] = 2
        imgs.append(img)
        generate_video(imgs)

        plt.figure()
        plt.imshow(img, cmap=cm.jet)
        plt.grid()
        plt.savefig(Path("process2_debugging") / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        plt.close()

    return best_box


def process_simple(img_np, bbox: np.ndarray, glyph: Image, cls: str, store_video: bool = False, padding=20) -> Image:
    if len(bbox) == 0:
        return

    padding_dicts = {'clef': (40, 31), 'accidental': (25, 15), 'notehead': (7, 7), 'key': (20, 15)}
    padding = padding_dicts[[key for key in padding_dicts.keys() if key in cls][0]]

    poly = rotated_box_to_poly_np(np.expand_dims(bbox, 0))[0]
    y_min, y_max = max(int(np.min(poly[1::2]) - padding[0]), 0), min(int(np.max(poly[1::2]) + padding[0]),
                                                                     img_np.shape[0])
    x_min, x_max = max(int(np.min(poly[::2]) - padding[1]), 0), min(int(np.max(poly[::2]) + padding[1]),
                                                                    img_np.shape[1])
    img_roi = img_np[y_min:y_max, x_min:x_max]
    #img_roi = 255 - img_roi  # invert

    best_box, best_overlap = None, -1
    glyph = GlyphGenerator()

    orig_angle = bbox[4]
    angles = _create_search_list(orig_angle - 0.2, orig_angle + 0.2, 0.05)

    method = eval('cv.TM_CCORR_NORMED')

    count_angle_not_improved = 0
    for angle in angles:

        proposed_glyph = glyph.get_transformed_glyph(cls, int(bbox[2]), int(bbox[3]), -angle, padding_left=None,
                                                     padding_right=None, padding_top=None, padding_bottom=None)

        img_roi_copy = img_roi.copy()
        try:
            res = cv.matchTemplate(img_roi_copy, proposed_glyph, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            top_left = max_loc
        except cv2.error as e:
            print(e)
            print(cls, img_roi_copy.shape, proposed_glyph.shape)
            max_val = -1

        if max_val > best_overlap:
            best_overlap = max_val
            center = (top_left[0] + proposed_glyph.shape[1] / 2, top_left[1] + proposed_glyph.shape[0] / 2)
            global_center = (x_min + center[0], y_min + center[1])
            best_box = tuple(global_center) + (int(bbox[2]), int(bbox[3]), angle)

        if count_angle_not_improved > 4:
            break

    return np.array(best_box)


def process(img: Image, bbox: np.ndarray, glyph: Image, cls: str) -> Image:
    if len(bbox) == 0:
        return

    img_np = np.array(img.convert('L')) < 128
    w, h = glyph.size
    x, y, _, _, _ = bbox
    w2 = w / 2
    h2 = h / 2
    img_roi = img_np[int(y - h2):int(y + h2), int(x - w2):int(x + w2)]

    mask_np = np.array(glyph) > 0
    mask_np = np.flipud(mask_np)
    y_pad, x_pad = img_roi.shape[0] - mask_np.shape[0], img_roi.shape[1] - mask_np.shape[1]
    mask_np = np.pad(mask_np, ((int(np.ceil(y_pad / 2)), y_pad // 2), (int(np.ceil(x_pad / 2)), x_pad // 2)))
    new_mask = optical_flow_merging(img_roi, mask_np)
    return PImage.fromarray(binary_erosion(new_mask))


def bbox_to_polygon(bbox: np.ndarray) -> Polygon:
    x, y, w, h, a = bbox
    w2 = w / 2.0
    h2 = h / 2.0
    p = Polygon([
        (x - w2, y - h2),
        (x + w2, y - h2),
        (x + w2, y + h2),
        (x - w2, y + h2),
    ])
    return rotate(p, a, use_radians=True)


def calc_loss(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    a = bbox_to_polygon(bbox1)
    b = bbox_to_polygon(bbox2)
    return a.intersection(b).area / a.union(b).area


def visualize(crop: Image, prop_bbox: np.ndarray, gt_bbox: np.ndarray,
              gt_glyph: Image, new_bbox: np.ndarray, new_glyph: Image):
    img = crop.copy().convert('RGBA')
    draw = Draw(img, 'RGBA')
    draw.polygon(list(bbox_to_polygon(prop_bbox).exterior.coords)[:4], outline='#E00')
    draw.polygon(list(bbox_to_polygon(gt_bbox).exterior.coords)[:4], outline='#1F2')
    draw.polygon(list(bbox_to_polygon(new_bbox).exterior.coords)[:4], outline='#EC0')
    # x, y, _, _, _ = prop_bbox
    # glyph_img = PImage.new('RGBA', img.size, '#0000')
    # Draw(glyph_img).bitmap((x - (gt_glyph.width // 2), y - (gt_glyph.height // 2)), gt_glyph, fill='#E005')
    # img = PImage.alpha_composite(img, glyph_img)
    x, y, _, _, _ = gt_bbox
    glyph_img = PImage.new('RGBA', img.size, '#0000')
    Draw(glyph_img).bitmap((x - (gt_glyph.width // 2), y - (gt_glyph.height // 2)), gt_glyph, fill='#1F28')
    img = PImage.alpha_composite(img, glyph_img)
    x, y, _, _, _ = new_bbox
    glyph_img = PImage.new('RGBA', img.size, '#0000')
    Draw(glyph_img).bitmap((x - (new_glyph.width // 2), y - (new_glyph.height // 2)), new_glyph, fill='#EC0A')
    img = PImage.alpha_composite(img, glyph_img)
    x, y, w, h, _ = gt_bbox
    img.crop((x - w // 2 - 50, y - h // 2 - 50, x + w // 2 + 50, y + h // 2 + 50)).show()


def _process_sample(img: Image, sample, whitelist=[]):
    scores = []
    det_bbox = sample['proposal']
    prop_bbox: np.ndarray = sample['proposal'][:5].astype(np.float)
    prop_bbox = bbox_translate(prop_bbox)

    cls: str = sample['proposal'][5]
    if len(whitelist) > 0:
        if len([x for x in whitelist if x in cls]) < 1:
            return sample['proposal'][:-1]

    if 'gt' in sample:
        gt_bbox: np.ndarray = sample['gt'][:5].astype(np.float)
        # print(f"IoU [{cls}]: ")
    for glyph in get_glyphs(cls, prop_bbox, 0):
        #new_glyph = process2(img, prop_bbox, glyph, cls)
        new_glyph = process_simple(img, prop_bbox, glyph, cls)

        derived_bbox = extract_bbox_from(glyph, prop_bbox, cls)

        if isinstance(new_glyph, np.ndarray):
            new_bbox = new_glyph
        else:
            new_bbox = extract_bbox_from(new_glyph, prop_bbox, cls)

        if 'gt' in sample:
            iou = calc_loss(gt_bbox, new_bbox)
            base_iou = calc_loss(gt_bbox, derived_bbox)
            # visualize(img, prop_bbox, gt_bbox, glyph, new_bbox, new_glyph)
            # print(f", {iou:.3} (baseline: {base_iou:.3})", end='')
            # print()
        return new_bbox


def _process_single(img: Image, samples, whitelist=[], n_workers=5):
    bboxes, jobs = [], []

    img = np.array(img.convert('L'))
    for sample in samples:
        bboxes.append(_process_sample(img, sample, whitelist))

    return bboxes


def process(samples, n_workers):
    for img_fp, boxes in samples:
        img = PImage.open(img_fp)
        _process_single(img, boxes, whitelist=['clef', 'accidental', 'notehead', 'key'], n_workers=n_workers)


if __name__ == '__main__':
    for i in range(1, 10):
        start = time.time()
        process(SAMPLES, i)
        print("Duration for workers=", i, ":", time.time() - start)