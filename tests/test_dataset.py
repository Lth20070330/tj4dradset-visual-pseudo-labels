import os
import unittest

import numpy as np
import torch

from radar_pseudo.dataset import TJ4DRadSet
from radar_pseudo.geometry import (
    points_in_kitti_box_camera,
    radar_to_rectified_camera,
    rectified_camera_to_radar,
)
from radar_pseudo.pseudo_label import backproject_pixel, densest_depth_cluster
from radar_pseudo.evaluate_teacher import box_iou_2d
from radar_pseudo.batch_generate import to_kitti_line
from radar_pseudo.pseudo_label import PseudoBox3D
from radar_pseudo.student import BEVConfig, RadarBEVDetector, RadarDetectionDataset, radar_to_bev
from radar_pseudo.metrics3d import ap_r40, bev_iou, iou_3d
from radar_pseudo.dataset import KittiObject
from radar_pseudo.filter_pseudo_labels import joint_center_score


SAMPLE_ROOT = os.environ.get("TJ4DRADSET_SAMPLE", r"C:\Users\lth\TJ4DRadSet\Sample")


class TJ4DRadSetSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = TJ4DRadSet(SAMPLE_ROOT)

    def test_sample_has_expected_complete_frames(self):
        self.assertEqual(len(self.dataset.frame_ids), 41)
        self.assertEqual(self.dataset.frame_ids[0], "070070")
        self.assertEqual(self.dataset.frame_ids[-1], "070110")

    def test_radar_has_eight_features_and_consistent_range(self):
        radar = self.dataset.load_radar("070070")
        self.assertEqual(radar.shape[1], 8)
        geometric_range = np.linalg.norm(radar[:, :3], axis=1)
        np.testing.assert_allclose(geometric_range, radar[:, 4], rtol=2e-4, atol=2e-4)

    def test_calibration_round_trip(self):
        calibration = self.dataset.load_calibration("070070")
        points = self.dataset.load_radar("070070")[:50, :3]
        camera = radar_to_rectified_camera(points, calibration)
        recovered = rectified_camera_to_radar(camera, calibration)
        np.testing.assert_allclose(recovered, points, rtol=1e-9, atol=1e-9)

    def test_labels_parse(self):
        labels = self.dataset.load_labels("070070")
        self.assertGreater(len(labels), 0)
        self.assertTrue(all(obj.dimensions_hwl.shape == (3,) for obj in labels))

    def test_point_box_association_returns_boolean_mask(self):
        radar = self.dataset.load_radar("070070")
        calibration = self.dataset.load_calibration("070070")
        points_camera = radar_to_rectified_camera(radar[:, :3], calibration)
        mask = points_in_kitti_box_camera(points_camera, self.dataset.load_labels("070070")[0])
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (len(radar),))
        self.assertGreater(mask.sum(), 0)

    def test_sample_has_no_images(self):
        self.assertFalse(self.dataset.has_image("070070"))

    def test_densest_depth_cluster_rejects_background(self):
        depths = np.array([43.5, 43.7, 44.0, 44.1, 44.2, 67.8, 78.7, 103.0])
        np.testing.assert_allclose(densest_depth_cluster(depths), [43.5, 43.7, 44.0, 44.1, 44.2])

    def test_backprojection_uses_camera_intrinsics(self):
        p2 = np.array([[1000, 0, 600, 0], [0, 1000, 400, 0], [0, 0, 1, 0]], dtype=float)
        np.testing.assert_allclose(backproject_pixel(700, 450, 20, p2), [2, 1, 20])

    def test_box_iou(self):
        self.assertEqual(box_iou_2d(np.array([0, 0, 2, 2]), np.array([0, 0, 2, 2])), 1.0)
        self.assertEqual(box_iou_2d(np.array([0, 0, 1, 1]), np.array([2, 2, 3, 3])), 0.0)

    def test_pseudo_kitti_line_has_fifteen_fields(self):
        box = PseudoBox3D(
            category="Car", dimensions_hwl=np.array([1.5, 1.8, 4.2]), location_camera=np.array([1, 2, 20]),
            rotation_y=-1.5, visual_confidence=.9, visual_depth_m=20, radar_depth_m=18,
            radar_points=4, quality=.8, position_source="radar_corrected", bbox_2d=np.array([10, 20, 30, 40]),
        )
        self.assertEqual(len(to_kitti_line(box).split()), 15)

    def test_bev_shape_and_model_output(self):
        radar = self.dataset.load_radar("070070")
        config = BEVConfig()
        features = radar_to_bev(radar, config)
        self.assertEqual(features.shape, (5, config.height, config.width))
        model = RadarBEVDetector()
        output = model(torch.from_numpy(features[None]))
        self.assertEqual(output["heatmap"].shape, (1, 5, config.height, config.width))
        self.assertEqual(output["regression"].shape, (1, 8, config.height, config.width))
        training_sample = RadarDetectionDataset(SAMPLE_ROOT, ["070070"])[0]
        self.assertEqual(training_sample["weight"].shape, (8, config.height, config.width))
        self.assertEqual(training_sample["positive_weight"].shape, (5, config.height, config.width))
        self.assertEqual(training_sample["classification_weight"].shape, (1, config.height, config.width))

    def test_oriented_box_iou_metrics(self):
        first = KittiObject(
            "Car", 0, 0, 0.0, np.zeros(4), np.array([2.0, 2.0, 4.0]), np.array([0.0, 1.0, 20.0]), 0.0
        )
        identical = KittiObject(
            "Car", 0, 0, 0.0, np.zeros(4), np.array([2.0, 2.0, 4.0]), np.array([0.0, 1.0, 20.0]), 0.0
        )
        disjoint = KittiObject(
            "Car", 0, 0, 0.0, np.zeros(4), np.array([2.0, 2.0, 4.0]), np.array([20.0, 1.0, 20.0]), 0.0
        )
        rotated = KittiObject(
            "Car", 0, 0, 0.0, np.zeros(4), np.array([2.0, 2.0, 4.0]), np.array([0.0, 1.0, 20.0]), np.pi / 2
        )
        self.assertAlmostEqual(bev_iou(first, identical), 1.0)
        self.assertAlmostEqual(iou_3d(first, identical), 1.0)
        self.assertEqual(bev_iou(first, disjoint), 0.0)
        self.assertAlmostEqual(bev_iou(first, rotated), 1.0 / 3.0)

    def test_ap_r40_perfect_ranking(self):
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 1.0])
        self.assertEqual(ap_r40(recall, precision), 1.0)

    def test_joint_center_score_is_geometric_mean(self):
        self.assertAlmostEqual(joint_center_score({"class_quality": 0.81, "center_quality": 0.49}), 0.63)


if __name__ == "__main__":
    unittest.main()
