import unittest

import cv2
import numpy as np

from plant_classifier.features import create_camera_leaf_mask


class FeatureMaskTests(unittest.TestCase):
    def test_leaf_mask_prefers_green_leaf_over_skin_background(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, :60] = [190, 140, 115]
        image[:, 60:] = [42, 135, 66]

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = create_camera_leaf_mask(hsv)

        skin_ratio = (mask[:, :60] > 0).mean()
        leaf_ratio = (mask[:, 60:] > 0).mean()

        self.assertLess(skin_ratio, 0.10)
        self.assertGreater(leaf_ratio, 0.85)


if __name__ == "__main__":
    unittest.main()
