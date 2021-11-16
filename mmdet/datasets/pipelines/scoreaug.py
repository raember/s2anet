from pathlib import Path
from typing import List, Dict

import PIL.Image
from numpy.random import choice
import numpy as np
from PIL.Image import Image, open as img_open
from PIL import Image as Image_m, ImageEnhance, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ..registry import PIPELINES

SEAMLESS = 'seamless'
SEAMED = 'seamed'

@PIPELINES.register_module
class ScoreAug(object):
    """
    Augment scores with real-world blank pages
    """
    _blank_pages_path: Path
    _seamless_imgs: List[str]
    _seamed_imgs: List[str]

    def __init__(self, blank_pages_path, padding_length = 200, p_blur=0.5):
        self._blank_pages_path = Path(blank_pages_path)
        assert self._blank_pages_path.exists(), "Path to blank pages must exist"
        assert self._blank_pages_path.is_dir(), "Path to blank pages must be a directory"
        self._seamless_imgs = self._load_images(self._blank_pages_path / SEAMLESS)
        self._seamed_imgs = self._load_images(self._blank_pages_path / SEAMED)
        self.padding_length = padding_length
        self.p_blur = p_blur



    def _load_images(self, path: Path) -> List[str]:
        assert path.exists(), f"Path to {path.name} blank pages must exist"
        assert path.is_dir(), f"Path to {path.name} blank pages must be a directory"

        return [str(i) for i in path.glob('*.png')]
        #return list(map(img_open, path.glob('*.png')))


    def __call__(self, results: dict):
        take_seamless = choice([True, False], p=[0.5, 0.5])
        if not take_seamless:
            bg_imgs = self._seamed_imgs
        else:
            bg_imgs = self._seamless_imgs


        # Random blank page background image
        bg_img = choice(bg_imgs)
        bg_img = Image_m.open(bg_img)
        shape = results['img_shape'][1::-1]
        bg_img = bg_img.resize(shape)

        # Random flips
        horiz_flip = choice([True, False], p=[0.5, 0.5])
        if horiz_flip:
            bg_img = bg_img.transpose(Image_m.FLIP_LEFT_RIGHT)
        vert_flip = choice([True, False], p=[0.5, 0.5])
        if vert_flip:
            bg_img = bg_img.transpose(Image_m.FLIP_TOP_BOTTOM)

        # maybe crop and resize
        crop_resize = choice([True, False], p=[0.2, 0.8])
        if crop_resize:
            crop_factor = np.random.uniform(low=0.25, high=0.85)
            crop_size = (crop_factor * np.array(shape)).astype(np.int32)
            max_topleft = np.array(shape) - crop_size
            top_left = np.random.uniform(low=[0, 0], high=max_topleft, size=2).astype(np.int32)
            bottom_right = top_left + crop_size
            bg_img = bg_img.crop(np.concatenate([top_left, bottom_right]))
            bg_img = bg_img.resize(shape)

        # Increase size if seamed
        if not (take_seamless or crop_resize):
            # compute new shape, resize background
            shape = tuple([x + self.padding_length for x in shape])
            bg_img = bg_img.resize(shape)

            # extend foreground
            img_extended = np.ones(shape[::-1] + (3,),dtype=np.uint8)*255
            half_pad = self.padding_length//2
            img_extended[half_pad:-half_pad, half_pad:-half_pad] = results['img']
            results['img'] = img_extended

            # shift bounding boxes
            results['ann_info']['bboxes'] = results['ann_info']['bboxes'] + half_pad
            results['gt_bboxes'] = results['gt_bboxes'] + half_pad

            # correct meta infos
            results['img_info']['width'] = shape[0]
            results['img_info']['height'] = shape[1]
            results['img_shape'] = shape[::-1]+(3,)

        # randomize bg brightness
        random_bg_brightness = choice([True, False], p=[0.5, 0.5])
        if random_bg_brightness or True:
            enhancer = ImageEnhance.Brightness(bg_img)
            bg_img = enhancer.enhance(np.random.uniform(0.8, 1.1))

        fg_img = Image_m.fromarray(results['img'])
        # high contrast contrast fg
        fg_high_contrast = choice([True, False], p=[0.2, 0.8])
        if fg_high_contrast or True:
            enhancer = ImageEnhance.Contrast(fg_img)
            fg_img = enhancer.enhance(5)

        # maybe add small rotation to score
        small_rotate = choice([True, False], p=[0.6, 0.4])
        if small_rotate:
            # negate angle to get teh right direction
            angle = -np.random.uniform(-2, 2)

            # Add some randomly dark bg to fill in
            fill = np.random.randint(5, 30)
            fill_col = tuple((fill + np.random.randint(-5, 5)) for _ in range(3))

            center = tuple(np.array(results['img'].shape[:2]) / 2)
            fg_img = fg_img.rotate(angle, PIL.Image.BICUBIC, center=center, fillcolor=fill_col)
            bg_img = bg_img.rotate(angle, PIL.Image.BICUBIC, center=center, fillcolor=fill_col)

            def rotate(arr: np.ndarray, angle: float) -> np.ndarray:
                ar = arr.copy()
                theta = np.radians(angle)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                for i, o_bbox in enumerate(ar):
                    bbox = o_bbox.reshape((4, 2)) - center
                    bbox = bbox.dot(R)
                    ar[i] = (bbox + center).reshape((8,))
                return ar

            results['ann_info']['bboxes'] = rotate(results['ann_info']['bboxes'], angle)
            results['gt_bboxes'] = rotate(results['gt_bboxes'], angle)

        fg_brightness = choice([True, False], p=[0.4, 0.6])
        if fg_brightness or True:
            fg_img = np.array(fg_img, dtype=np.uint32)
            fg_img = fg_img + np.random.uniform(20, 90)
            fg_img[fg_img > 255] = 255
            fg_img = Image_m.fromarray(fg_img.astype(np.uint8))


        fg_blur = choice([True, False], p=[self.p_blur, 1-self.p_blur])
        if fg_blur or True:
            fg_img = fg_img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(1, 2)))


        # Merge
        results['img'] = np.minimum(fg_img, bg_img)
        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(20, 30))
        # plt.imshow(results['img'], interpolation='nearest')
        # plt.show()
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(blank_pages_path={self._blank_pages_path})'
