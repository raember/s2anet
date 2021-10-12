from pathlib import Path
from typing import List
from numpy.random import choice, r
import numpy as np
from PIL.Image import Image, open as img_open
from PIL import Image as Image_m

from ..registry import PIPELINES

SEAMLESS = 'seamless'
SEAMED = 'seamed'

@PIPELINES.register_module
class ScoreAug(object):
    """
    Augment scores with real-world blank pages
    """
    _blank_pages_path: Path
    _seamless_imgs: List[Image]
    _seamed_imgs: List[Image]

    def __init__(self, blank_pages_path):
        self._blank_pages_path = Path(blank_pages_path)
        assert self._blank_pages_path.exists(), "Path to blank pages must exist"
        assert self._blank_pages_path.is_dir(), "Path to blank pages must be a directory"
        self._seamless_imgs = self._load_images(self._blank_pages_path / SEAMLESS)
        self._seamed_imgs = self._load_images(self._blank_pages_path / SEAMED)


    @staticmethod
    def _load_images(path: Path) -> List[Image]:
        assert path.exists(), f"Path to {path.name} blank pages must exist"
        assert path.is_dir(), f"Path to {path.name} blank pages must be a directory"
        return list(map(img_open, path.glob('*.png')))


    def __call__(self, results: dict):
        take_seamless = choice([True, False])
        if not take_seamless:
            bg_imgs = self._seamed_imgs
            RESIZE_MAX_MIN = (0.85, 0.7)
        else:
            bg_imgs = self._seamless_imgs
            RESIZE_MAX_MIN = (1.0, 0.9)

        # Random blank page background image
        bg_img: Image = choice(bg_imgs)

        # Random flips
        horiz_flip = choice([True, False])
        if horiz_flip:
            bg_img = bg_img.transpose(Image_m.FLIP_LEFT_RIGHT)
        vert_flip = choice([True, False])
        if vert_flip:
            bg_img = bg_img.transpose(Image_m.FLIP_TOP_BOTTOM)

        #TODO: Add resizing of source image and bboxes

        # Merge
        #TODO: Fix results['img']
        bg = np.array(bg_img.resize(results['img'].shape[:2][::-1]))
        results['img'] = np.minimum(results['img'], bg)


    def __repr__(self):
        return f'{self.__class__.__name__}(blank_pages_path={self._blank_pages_path})'
