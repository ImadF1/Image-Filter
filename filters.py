from __future__ import annotations

import cv2
import numpy as np

ImageArray = np.ndarray


def _to_uint8(image: ImageArray) -> ImageArray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_rgb(image: ImageArray) -> ImageArray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def adjust_brightness_contrast(
    image: ImageArray,
    brightness: int = 0,
    contrast: int = 0,
) -> ImageArray:
    rgb_image = _to_rgb(image).astype(np.float32)
    alpha = 1.0 + (float(contrast) / 100.0)
    adjusted = (rgb_image - 127.5) * alpha + 127.5 + float(brightness)
    return _to_uint8(adjusted)


def apply_grayscale(image: ImageArray) -> ImageArray:
    gray = cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_gaussian_blur(
    image: ImageArray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> ImageArray:
    rgb_image = _to_rgb(image)
    size = _odd_kernel_size(kernel_size)
    return cv2.GaussianBlur(rgb_image, (size, size), sigmaX=float(sigma))


def apply_edge_detection(
    image: ImageArray,
    threshold1: int = 100,
    threshold2: int = 200,
) -> ImageArray:
    gray = cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(threshold1), int(threshold2))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_sepia(image: ImageArray, intensity: float = 1.0) -> ImageArray:
    rgb_image = _to_rgb(image).astype(np.float32)
    amount = float(np.clip(intensity, 0.0, 1.0))
    sepia_matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    transformed = rgb_image @ sepia_matrix.T
    blended = rgb_image * (1.0 - amount) + transformed * amount
    return _to_uint8(blended)


def apply_vintage(image: ImageArray, intensity: float = 0.6) -> ImageArray:
    rgb_image = _to_rgb(image)
    amount = float(np.clip(intensity, 0.0, 1.0))
    warmed = apply_sepia(rgb_image, 0.35 + (0.45 * amount))

    hsv_image = cv2.cvtColor(warmed, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[..., 1] *= 1.0 - (0.35 * amount)
    hsv_image[..., 2] *= 1.0 - (0.08 * amount)
    faded = cv2.cvtColor(_to_uint8(hsv_image), cv2.COLOR_HSV2RGB).astype(np.float32)

    height, width = faded.shape[:2]
    x_kernel = cv2.getGaussianKernel(width, max(width / 2.2, 1.0))
    y_kernel = cv2.getGaussianKernel(height, max(height / 2.2, 1.0))
    mask = y_kernel @ x_kernel.T
    mask = mask / mask.max()
    vignette = 1.0 - ((1.0 - mask) * (0.4 * amount))

    vintage = faded * vignette[..., None] + (8.0 * amount)
    return _to_uint8(vintage)


def overlay_edges(
    image: ImageArray,
    edges: ImageArray,
    color: tuple[int, int, int] = (255, 255, 255),
) -> ImageArray:
    rgb_image = _to_rgb(image).copy()
    edge_mask = _to_rgb(edges)[..., 0] > 0
    rgb_image[edge_mask] = np.array(color, dtype=np.uint8)
    return rgb_image
