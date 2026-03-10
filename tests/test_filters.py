import unittest

import numpy as np

from filters import (
    adjust_brightness_contrast,
    apply_edge_detection,
    apply_gaussian_blur,
    apply_grayscale,
    apply_sepia,
    apply_vintage,
    overlay_edges,
)


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        x = np.linspace(0, 255, 48, dtype=np.uint8)
        y = np.linspace(255, 0, 32, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        self.image = np.stack(
            [xv, yv, np.full_like(xv, 128)],
            axis=-1,
        )

    def assert_valid_image(self, image: np.ndarray) -> None:
        self.assertEqual(image.shape, self.image.shape)
        self.assertEqual(image.dtype, np.uint8)

    def test_grayscale_keeps_three_channels(self) -> None:
        result = apply_grayscale(self.image)
        self.assert_valid_image(result)
        self.assertTrue(np.array_equal(result[..., 0], result[..., 1]))
        self.assertTrue(np.array_equal(result[..., 1], result[..., 2]))

    def test_sepia_changes_pixels(self) -> None:
        result = apply_sepia(self.image, intensity=0.8)
        self.assert_valid_image(result)
        self.assertFalse(np.array_equal(result, self.image))

    def test_vintage_changes_pixels(self) -> None:
        result = apply_vintage(self.image, intensity=0.7)
        self.assert_valid_image(result)
        self.assertFalse(np.array_equal(result, self.image))

    def test_blur_preserves_shape(self) -> None:
        result = apply_gaussian_blur(self.image, kernel_size=8, sigma=1.2)
        self.assert_valid_image(result)

    def test_edge_detection_preserves_shape(self) -> None:
        result = apply_edge_detection(self.image, threshold1=40, threshold2=120)
        self.assert_valid_image(result)

    def test_overlay_edges_preserves_shape(self) -> None:
        edges = apply_edge_detection(self.image)
        result = overlay_edges(self.image, edges)
        self.assert_valid_image(result)

    def test_brightness_and_contrast_preserve_shape(self) -> None:
        result = adjust_brightness_contrast(self.image, brightness=20, contrast=25)
        self.assert_valid_image(result)


if __name__ == "__main__":
    unittest.main()
