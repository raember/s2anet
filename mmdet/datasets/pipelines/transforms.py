import inspect
from typing import Tuple, Optional, List, Union

import albumentations
import mmcv
import numpy as np
from albumentations import Compose
from imagecorruptions import corrupt
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES


@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale) or a single scalar.
    There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple] or scalar): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 max_size=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple) or mmcv.is_list_of(self.img_scale, float)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.max_size = max_size

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):

        resize_target = results['scale']
        if self.max_size is not None:
            if results['img'].shape[1] > self.max_size[0] or \
                    results['img'].shape[0] > self.max_size[1]:
                resize_target = self.max_size

        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], resize_target, return_scale=True)

        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], resize_target, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if bboxes.shape[0] != 0:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            # padded_img = mmcv.impad(results['img'], self.size)
            padded_img = mmcv.impad(results['img'], shape=(1024, 1024), pad_val=0)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


class OBBox:
    """Toolbox for handling oriented bboxes"""
    @staticmethod
    def get_inside_outside_edge_mask(bboxes: np.ndarray, crop_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate which bounding boxes are completely inside, completely outside or intersecting a given shape.

        Edge cases can be bboxes that have 0-3 corners inside the boundary.

        :param crop_shape: Shape of the crop.
        :type crop_shape: Tuple[int]
        :param bboxes: List of 8 dimensional bounding boxes.
        :type bboxes: np.ndarray
        :return: A tuple containing boolean masks for the corresponding bbox being inside, outside or an edge case.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        assert bboxes.shape[1] == 8
        one_dim_bbox_shape = bboxes[:, 0::2].shape
        one_dim_ones = np.ones(one_dim_bbox_shape)
        x = one_dim_ones * crop_shape[0]
        y = one_dim_ones * crop_shape[1]
        zeros = np.zeros(one_dim_bbox_shape)
        # Find which coords are outside the crop.
        # If sum = 4, all corners are too far away.
        # If sum = 0, the corners are within the particular boundary condition.
        # If sum in [1, 2, 3], it might be an edge case.
        too_high = (bboxes[:, 0::2] < zeros).sum(axis=1)
        too_low = (bboxes[:, 0::2] > x).sum(axis=1)
        too_left = (bboxes[:, 1::2] < zeros).sum(axis=1)
        too_right = (bboxes[:, 1::2] > y).sum(axis=1)
        # Bbox is outside if the corners are outside in any direction
        outside = ((too_high == 4) + (too_low == 4) + (too_left == 4) + (too_right == 4)) == 1
        # Bbox is inside if the corners are inside in all directions
        inside = (too_high == 0) * (too_low == 0) * (too_left == 0) * (too_right == 0)
        # If the bbox is neither completely inside nor outside, it's an edge case and needs to be handles separately
        edge_cases = ((1 - outside) + (1 - inside)) == 2
        return inside, outside, edge_cases

    @staticmethod
    def is_point_inside(point: np.ndarray, crop_shape: Tuple[int, int]) -> bool:
        """
        Test whether a point is inside a crop.

        Regard points on the border to be inside as well.

        :param point: The point to test.
        :type point: np.ndarray
        :param crop_shape: The crop shape to test for.
        :type crop_shape: Tuple[int]
        :return: True if the point is inside the crop. False otherwise.
        :rtype: bool
        """
        assert point.shape == (2,)
        x, y = point
        if x < 0 or x > crop_shape[0]:
            return False
        if y < 0 or y > crop_shape[1]:
            return False
        return True

    @staticmethod
    def is_bbox_inside(bbox: Union[List[Tuple[float, float]], np.ndarray], crop_shape: Tuple[int, int]) -> bool:
        """
        Test whether a bounding box is inside a crop.

        Regard points on the border to be inside as well.

        :param bbox: The coordinates of the bbox.
        :type bbox: Union[List[Tuple[float, float]], np.ndarray]
        :param crop_shape: The crop shape to test for.
        :type crop_shape: Tuple[int]
        :return: True if bbox is inside the crop. False otherwise.
        :rtype: bool
        """
        if isinstance(bbox, np.ndarray) and bbox.shape != (4, 2):
            bbox = bbox.copy().reshape((4, 2))
        for point in bbox:
            if not OBBox.is_point_inside(point, crop_shape):
                return False
        return True

    @staticmethod
    def crop_bbox(
            corners: np.ndarray,
            crop_shape: Tuple[int, int],
            threshold_rel: float = 0.6,
            threshold_abs: float = 20.0) -> Optional[np.ndarray]:
        """
        Crop bounding box along a given shape.

        Find the intersecting edges and shorten them so the rectangle fits inside the border.

        :param corners: The corners of the bbox.
        :type corners: np.ndarray
        :param crop_shape: The border shape of the image.
        :type crop_shape: Tuple[int]
        :param threshold_rel: The threshold of the new area divided by the old area under which to dismiss bboxes.
        :type threshold_rel: float
        :param threshold_abs: The threshold of the new area over which to accept crops even if they'e below the relative threshold.
        :type threshold_abs: float
        :return: The cropped bbox.
        :rtype: Optional[np.ndarray]
        """
        assert corners.shape == (4, 2)
        if OBBox.is_bbox_inside(corners, crop_shape):
            return corners
        orig_area = OBBox.get_area(corners)
        bboxes_by_area = {}
        for bbox in OBBox._get_possible_crop_solutions(corners, crop_shape):
            if not OBBox.is_bbox_inside(bbox, crop_shape):
                continue
            area = OBBox.get_area(bbox)
            if area > orig_area * threshold_rel or area > threshold_abs:
                bboxes_by_area[area] = bbox
        if len(bboxes_by_area) == 0:
            # No suitable crop available
            return None
        highest_area = max(bboxes_by_area.keys())
        return bboxes_by_area[highest_area]

    @staticmethod
    def _get_possible_crop_solutions(corners: np.ndarray, crop_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Find possible crops that lie inside the crop boundary

        :param corners: The corners of the bbox
        :type corners: np.ndarray
        :param crop_shape: The shape of the crop boundary
        :type crop_shape: Tuple[int, int]
        :return: A list of all possible crops
        :rtype: List[np.ndarray]
        """
        return OBBox._get_possible_side_truncated_crops(corners, crop_shape) + \
               OBBox._get_possible_diagonal_crops(corners, crop_shape)

    @staticmethod
    def _get_possible_diagonal_crops(corners: np.ndarray, crop_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Find possible crops that lie inside the crop boundary by truncating along the diagonal

        :param corners: The corners of the bbox
        :type corners: np.ndarray
        :param crop_shape: The shape of the crop
        :type crop_shape: Tuple[int, int]
        :return: A list of possible crops
        :rtype: List[np.ndarray]
        """
        truncated = []
        for i in range(4):
            corner = corners[i]
            if not OBBox.is_point_inside(corner, crop_shape):
                continue
            forward_idx = (i + 1) % 4
            backward_idx = (i - 1) % 4
            opposite_idx = (i + 2) % 4
            opposite = corners[opposite_idx]
            if not OBBox.is_point_inside(opposite, crop_shape):
                intersec = OBBox._intersect(corner, opposite, crop_shape)
                if intersec is None:
                    continue
                new_corners = corners.copy()
                new_corners[opposite_idx] = intersec
                # Get intersections of the edges shifted to the new opposite corner
                forward_ext = intersec - (opposite - corners[forward_idx])
                backward_ext = intersec - (opposite - corners[backward_idx])
                new_corners[forward_idx] = OBBox._seg_intersect(corner, corners[forward_idx], intersec, forward_ext)
                new_corners[backward_idx] = OBBox._seg_intersect(corner, corners[backward_idx], intersec, backward_ext)
                truncated.append(new_corners)
        for i, bbox in enumerate(truncated):
            if not OBBox.is_bbox_inside(bbox, crop_shape):
                # Although this is not a solution, recurse to possibly find a better solution based off of it
                truncated.pop(i)
                # Recurse
                truncated += OBBox._get_possible_crop_solutions(bbox, crop_shape)
        return truncated

    @staticmethod
    def _get_possible_side_truncated_crops(corners: np.ndarray, crop_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Find possible crops that lie inside the crop boundary by truncating along the edges.

        :param corners: The corners of the bbox
        :type corners: np.ndarray
        :param crop_shape: The shape of the crop
        :type crop_shape: Tuple[int, int]
        :return: A list of possible crops
        :rtype: List[np.ndarray]
        """
        truncated = []
        # Identify which edge crosses the crop border to shorten them
        for i in range(4):
            corner = corners[i]
            forward_idx = (i + 1) % 4
            backward_idx = (i - 1) % 4
            opposite_idx = (i + 2) % 4

            forward_intersection = OBBox._intersect(corner, corners[forward_idx], crop_shape)
            if forward_intersection is not None:
                # The current corner and the next corner are separated by the crop border
                if OBBox.is_point_inside(corner, crop_shape):
                    # The current corner is inside
                    # The next corner along with the corner opposite of it need to be shifted
                    retract_idx = forward_idx
                    adj_idx = opposite_idx
                else:
                    # The current corner is outside
                    # Itself and the corner before it need to be shifted
                    retract_idx = i
                    adj_idx = backward_idx
                new_bbox = OBBox._retract_to_intersection(corners.copy(), forward_intersection, retract_idx, adj_idx)
                is_new = True
                for bbox in truncated:
                    is_new &= not np.allclose(bbox, new_bbox)
                if is_new:
                    truncated.append(new_bbox)

            backward_intersection = OBBox._intersect(corner, corners[backward_idx], crop_shape)
            if backward_intersection is not None:
                # The current corner and the last corner are separated by the crop border
                if OBBox.is_point_inside(corner, crop_shape):
                    # The current corner is inside
                    # The last corner along with the corner opposite of it need to be shifted
                    retract_idx = backward_idx
                    adj_idx = opposite_idx
                else:
                    # The current corner is outside
                    # Itself and the corner after it need to be shifted
                    retract_idx = i
                    adj_idx = forward_idx
                new_bbox = OBBox._retract_to_intersection(corners.copy(), backward_intersection, retract_idx, adj_idx)
                is_new = True
                for bbox in truncated:
                    is_new &= not np.allclose(bbox, new_bbox)
                if is_new:
                    truncated.append(new_bbox)
        for i, bbox in enumerate(truncated):
            if not OBBox.is_bbox_inside(bbox, crop_shape):
                # Although this is not a solution, recurse to possibly find a better solution based off of it
                truncated.pop(i)
                # Recurse
                truncated += OBBox._get_possible_crop_solutions(bbox, crop_shape)
        return truncated

    # noinspection DuplicatedCode
    @staticmethod
    def _retract_one_side(corners: np.ndarray, crop_shape: Tuple[int, int]) -> np.ndarray:
        """
        Search for one side to be contracted as to make the bbox fit inside the boundary.

        Make sure to return the retraction with the lowest area loss.

        :param corners: The corners of the bbox
        :type corners: np.ndarray
        :param crop_shape: The shape of the image boundary
        :type crop_shape: Tuple[int, int]
        :return: A bbox with one side corrected
        :rtype: np.ndarray
        """
        assert corners.shape == (4, 2)
        if OBBox.is_bbox_inside(corners, crop_shape):
            return corners
        retraction_candidates = {}
        # Identify which edge crosses the crop border to shorten them
        for i in range(4):
            corner = corners[i]
            forward_idx = (i + 1) % 4
            backward_idx = (i - 1) % 4
            opposite_idx = (i + 2) % 4

            forward_intersection = OBBox._intersect(corner, corners[forward_idx], crop_shape)
            if forward_intersection is not None:
                if OBBox.is_point_inside(corner, crop_shape):
                    retract_idx = forward_idx
                    adj_idx = opposite_idx
                else:
                    retract_idx = i
                    adj_idx = backward_idx
                new_corners = OBBox._retract_to_intersection(corners.copy(), forward_intersection, retract_idx, adj_idx)
                retraction_candidates[OBBox.get_area(new_corners)] = new_corners

            backward_intersection = OBBox._intersect(corner, corners[backward_idx], crop_shape)
            if backward_intersection is not None:
                if OBBox.is_point_inside(corner, crop_shape):
                    retract_idx = backward_idx
                    adj_idx = opposite_idx
                else:
                    retract_idx = i
                    adj_idx = forward_idx
                new_corners = OBBox._retract_to_intersection(corners.copy(), backward_intersection, retract_idx, adj_idx)
                retraction_candidates[OBBox.get_area(new_corners)] = new_corners
        if len(retraction_candidates) == 0:
            # No side left to retract
            return corners
        highest_area = max(retraction_candidates.keys())
        return retraction_candidates[highest_area]

    @staticmethod
    def _retract_to_intersection(corners: np.ndarray, intersec: np.ndarray, corner_idx: int, adj_corner_idx: int) -> np.ndarray:
        """
        Retract corners from indexed croner to intersection.

        :param corners: The corners of the bbox
        :type corners: np.ndarray
        :param intersec: The intersection to retract to
        :type intersec: np.ndarray
        :param corner_idx: The index or the corner from which to retract from
        :type corner_idx: int
        :param adj_corner_idx: The index or the adjacent corner to include in the retraction
        :type adj_corner_idx: int
        :return: The resulting bbox
        :rtype: np.ndarray
        """
        offset = intersec - corners[corner_idx]
        corners[corner_idx] += offset
        corners[adj_corner_idx] += offset
        return corners

    @staticmethod
    def get_area(corners: np.ndarray) -> float:
        """
        Calculate the area of a bounding box based on its corners.

        Assumption of a rectangle is acceptable, as the area is only used for comparisons.

        :param corners: The four corners of the bbox.
        :type corners: np.ndarray
        :return: The area encased by the bbox corners
        :rtype: float
        """
        assert corners.shape == (4, 2)
        w = np.linalg.norm(corners[1] - corners[0])
        h = np.linalg.norm(corners[3] - corners[0])
        return w * h

    @staticmethod
    def get_angle(corners: np.ndarray) -> float:
        assert corners.shape == (4, 2)
        a, b, _, d = corners
        if np.linalg.norm(b - a) > np.linalg.norm(d - a):  # The longer edge provides a better measurement for the angle
            v = b - a
        else:
            v = d - a
        v = v/np.sqrt(np.dot(v, v))
        return np.rad2deg(np.arccos(v))[0]

    @staticmethod
    def get_edge_ratio(corners: np.ndarray) -> float:
        assert corners.shape == (4, 2)
        a, b, _, d = corners
        v1 = b - a
        v2 = d - a
        return np.linalg.norm(v1)/np.linalg.norm(v2)

    @staticmethod
    def _intersect(start_corner: np.ndarray, end_corner: np.ndarray, crop_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Find the intersection of the given edge and the border closest to the first corner.

        :param start_corner: The corner of the edge that is inside
        :type start_corner: np.ndarray
        :param end_corner: The corner of the edge that is outside
        :type end_corner: np.ndarray
        :param crop_shape: The shape that defines the border
        :type crop_shape: Tuple[int, int]
        :return: The intersecting corner of the given edge and the border
        :rtype: np.ndarray
        """
        assert start_corner.shape == (2,)
        assert end_corner.shape == (2,)
        if OBBox.is_point_inside(start_corner, crop_shape) and OBBox.is_point_inside(end_corner, crop_shape):
            return None
        edge_vec = end_corner - start_corner
        max_dist = np.float32(np.linalg.norm(edge_vec))
        edge_unit_vec = edge_vec/np.linalg.norm(edge_vec)
        intersecs = {}
        # In case the dimensions are mixed up, fix here:
        border_corners = [
            np.array((0, 0)),
            np.array((crop_shape[0], 0)),
            np.array((crop_shape[0], crop_shape[1])),
            np.array((0, crop_shape[1])),
        ]
        for i in range(4):
            border_a = border_corners[i]
            border_b = border_corners[(i + 1) % 4]
            intersection = OBBox._seg_intersect(start_corner, end_corner, border_a, border_b)
            if intersection is None:  # If the edge and the border are parallel, there's no intersection
                continue
            if min(border_a[0], border_b[0]) <= intersection[0] <= max(border_a[0], border_b[0]) and \
                min(border_a[1], border_b[1]) <= intersection[1] <= max(border_a[1], border_b[1]):
                if (intersection == start_corner).all() or (intersection == end_corner).all():
                    # Do no pursue the path of total despair, for only the darkest monsters dwell within those crevices
                    continue
                offset_vec = intersection - start_corner
                if offset_vec[0] == 0.0 and offset_vec[1] == 0.0:
                    # Intersection and corner coincide
                    continue
                dist = np.float32(np.linalg.norm(offset_vec))  # Floaty bois be floaty
                offset_unit_vec = offset_vec/dist
                if 0 < dist < max_dist and np.allclose(offset_unit_vec, edge_unit_vec):
                    intersecs[dist] = intersection
        if len(intersecs) == 0:
            return None
        return intersecs[min(intersecs.keys())]

    @staticmethod
    def _seg_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate the intersection point of two lines.

        https://stackoverflow.com/questions/3252194/numpy-and-line-intersections#3252222

        :param a1: Start point of the first line
        :type a1: np.ndarray
        :param a2: End point of the first line
        :type a2: np.ndarray
        :param b1: Start point of the second line
        :type b1: np.ndarray
        :param b2: End point of the second line
        :type b2: np.ndarray
        :return: Intersection point of found
        :rtype: Optional[np.ndarray]
        """
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = np.empty_like(da)
        dap[0] = -da[1]
        dap[1] = da[0]
        denom = np.dot(dap, db)
        if denom == 0.0:
            return None
        num = np.dot(dap, dp)
        return (num / denom.astype(float)) * db + b1



@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any bbox area. Default to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    Note:
        - If the image is smaller than the crop size, return the original image
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 allow_negative_crop=False,
                 bbox_clip_border=True,
                 threshold_rel=0.6,
                 threshold_abs=20.0):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.threshold_rel = threshold_rel
        self.threshold_abs = threshold_abs
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h, offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                inside, outside, edge_cases = OBBox.get_inside_outside_edge_mask(bboxes, self.crop_size)
                for i, (o_bbox, is_edge_case) in enumerate(zip(bboxes, edge_cases)):
                    if not is_edge_case:
                        continue
                    corners = o_bbox.reshape((4, 2))

                    # Crop bbox
                    cropped_bbox = OBBox.crop_bbox(
                        corners,
                        self.crop_size,
                        threshold_rel=self.threshold_rel,
                        threshold_abs=self.threshold_abs,
                    )
                    if cropped_bbox is None:
                        # The cropped bbox is faulty (i.e. too small)
                        # Treat as if outside as to discard it
                        edge_cases[i] = False
                        outside[i] = True
                    else:
                        bboxes[i] = cropped_bbox.reshape((8,))
            assert bboxes.shape[0] == inside.sum() + outside.sum() + edge_cases.sum()
            valid_inds = inside | edge_cases
            # If the crop does not contain any gt-bbox area and
            # self.allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}), '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module
class SegResizeFlipPadRescale(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. pad
    4. rescale (so that the final size can be different from the image size)

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        if results['keep_ratio']:
            gt_seg = mmcv.imrescale(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        if results['flip']:
            gt_seg = mmcv.imflip(gt_seg)
        if gt_seg.shape != results['pad_shape']:
            gt_seg = mmcv.impad(gt_seg, results['pad_shape'][:2])
        if self.scale_factor != 1:
            gt_seg = mmcv.imrescale(
                gt_seg, self.scale_factor, interpolation='nearest')
        results['gt_semantic_seg'] = gt_seg
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)


@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label

    def __call__(self, results):
        if random.randint(2):
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = expand_gt_masks

        # not tested
        if 'gt_semantic_seg' in results:
            assert self.seg_ignore_label is not None
            gt_seg = results['gt_semantic_seg']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label).astype(gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_semantic_seg'] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={}, ' \
                    'seg_ignore_label={})'.format(
                        self.mean, self.to_rgb, self.ratio_range,
                        self.seg_ignore_label)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) *
                        (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = [
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ]

                # not tested
                if 'gt_semantic_seg' in results:
                    results['gt_semantic_seg'] = results['gt_semantic_seg'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruption, self.severity)
        return repr_str


@PIPELINES.register_module
class Albu(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = [
                        results['masks'][i] for i in results['idx_mapper']
                    ]

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str