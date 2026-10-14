"""Basic smoke tests that do not ship the original portrait."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from portrait_grade import ellipse, process


class PortraitGradeTests(unittest.TestCase):
    def test_ellipse_mask_dimensions(self):
        mask = ellipse(64, 96, 0.5, 0.5, 0.2, 0.2)
        self.assertEqual(mask.shape, (64, 96))
        self.assertGreater(mask.max(), 0.9)

    def test_keeps_image_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.png')
            output = os.path.join(directory, 'retouched.jpg')
            img = np.full((120, 90, 3), 210, dtype=np.uint8)
            cv2.rectangle(img, (12, 55), (75, 110), (110, 130, 160), -1)
            self.assertTrue(cv2.imwrite(source, img))
            process(source, output, intensity=0.8)
            result = cv2.imread(output)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, img.shape)


if __name__ == '__main__':
    unittest.main()
