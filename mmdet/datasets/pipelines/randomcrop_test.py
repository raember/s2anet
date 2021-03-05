import unittest
import numpy as np
from mmdet.datasets.pipelines.transforms import OBBox, RandomCrop


class RandomCropTest(unittest.TestCase):
    BOUNDARY = (10, 10)
    def test_orthogonal_bbox_all_corners_inside(self):
        bbox = np.array((
            (3, 3),
            (5, 3),
            (5, 7),
            (3, 7),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY), bbox)

    def test_orthogonal_bbox_corners_inside_but_touching_side_of_boundary(self):
        bbox = np.array((
            (0, 3),
            (5, 3),
            (5, 7),
            (0, 7),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY), bbox)

    def test_orthogonal_bbox_corners_inside_but_touching_corner_of_boundary(self):
        bbox = np.array((
            (0, 3),
            (5, 3),
            (5, 10),
            (0, 10),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY), bbox)

    def test_orthogonal_bbox_two_corners_out(self):
        bbox = np.array((
            (3, 3),
            (11, 3),
            (11, 7),
            (3, 7),
        ), dtype='float64')
        new_bbox = np.array((
            (3, 3),
            (10, 3),
            (10, 7),
            (3, 7),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.8), new_bbox)

    def test_orthogonal_bbox_two_corners_out_high_threshold(self):
        bbox = np.array((
            (3, 3),
            (11, 3),
            (11, 7),
            (3, 7),
        ), dtype='float64')
        self.assertIsNone(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.9))

    def test_orthogonal_bbox_three_corners_out(self):
        bbox = np.array((
            (3, -2),
            (11, -2),
            (11, 7),
            (3, 7),
        ), dtype='float64')
        new_bbox = np.array((
            (3, 0),
            (10, 0),
            (10, 7),
            (3, 7),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.6), new_bbox)

    def test_orthogonal_bbox_three_corners_out_high_threshold(self):
        bbox = np.array((
            (3, -2),
            (11, -2),
            (11, 7),
            (3, 7),
        ), dtype='float64')
        self.assertIsNone(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.7))

    def test_orthogonal_bbox_four_corners_out(self):
        bbox = np.array((
            (-1, 2),
            (11, 2),
            (11, 7),
            (-1, 7),
        ), dtype='float64')
        new_bbox = np.array((
            (0, 2),
            (10, 2),
            (10, 7),
            (0, 7),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.5), new_bbox)

    def test_oriented_bbox_all_corners_inside(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY), bbox)

    def test_oriented_bbox_all_corners_inside_one_touching(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 1.0
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY), bbox)

    def test_oriented_bbox_all_corners_inside_two_touching(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 1.0
        bbox[:,0] += 1.0
        np.testing.assert_equal(OBBox.crop_bbox(bbox, self.BOUNDARY), bbox)

    def test_oriented_bbox_one_outside(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 2.0
        new_bbox = np.array((
            (2.3333333, 2),
            (8.3333333, 0),
            (9, 2),
            (3, 4),
        ), dtype='float64')
        np.testing.assert_almost_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.6), new_bbox)

    def test_oriented_bbox_one_outside_high_threshold(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 2.0
        self.assertIsNone(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.7))

    def test_oriented_bbox_two_outside(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 2.0
        bbox[:,0] += 2.0
        new_bbox = np.array((
            (4.2222222, 1.6666666),
            (9.2222222, 0),
            (10, 2.3333333),
            (5, 4),
        ), dtype='float64')
        np.testing.assert_almost_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.6), new_bbox)

    def test_oriented_bbox_two_outside_one_touching(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 3.0
        bbox[:,0] += 2.0
        new_bbox = np.array((
            (4.5555555, 1.6666666),
            (9.5555555, 0),
            (10, 1.3333333),
            (5, 3),
        ), dtype='float64')
        np.testing.assert_almost_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.3), new_bbox)

    def test_oriented_bbox_three_outside(self):
        bbox = np.array((
            (2, 3),
            (8, 1),
            (9, 4),
            (3, 6),
        ), dtype='float64')
        bbox[:,1] -= 4.0
        bbox[:,0] += 2.0
        new_bbox = np.array((
            (4.8888888, 1.6666666),
            (9.8888888, 0),
            (10, 0.3333333),
            (5, 2),
        ), dtype='float64')
        np.testing.assert_almost_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.0), new_bbox)

    def test_oriented_bbox_four_outside(self):
        bbox = np.array((
            (-2, 7),
            (11, 4),
            (12, 7),
            (-1, 10),
        ), dtype='float64')
        new_bbox = np.array((
            (0, 6.5384615),
            (9, 4.4615384),
            (10, 7.4615384),
            (1, 9.5384615),
        ), dtype='float64')
        np.testing.assert_almost_equal(OBBox.crop_bbox(bbox, self.BOUNDARY, threshold_rel=0.5), new_bbox)


if __name__ == '__main__':
    unittest.main()
