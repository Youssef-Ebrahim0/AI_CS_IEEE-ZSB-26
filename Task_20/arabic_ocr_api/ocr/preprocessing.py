"""
Image preprocessing for inference. Must mirror preprocess_image() from the
training notebook exactly -- decode -> float32 [0,1] -> resize to (H, W) ->
transpose so width becomes the CTC time dimension. Any deviation here
(resize order, normalization range, transpose) will silently produce wrong
predictions, since the model was trained on this exact pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def _decode_with_opencv(image_bytes: bytes) -> tf.Tensor:
    """Fallback decoder for image bytes tf.image.decode_image can't handle
    (some BMP/TIFF variants, corrupted headers, etc). Decodes straight to
    grayscale, matching channels=1 used everywhere else in the pipeline.
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("OpenCV could not decode the image bytes.")
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    return tf.convert_to_tensor(img)


def preprocess_image_bytes(image_bytes: bytes, img_height: int, img_width: int) -> tf.Tensor:
    """Returns a (img_width, img_height, 1) float32 tensor ready to be
    batched and fed to the model, exactly matching the training pipeline.
    """
    try:
        image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
    except Exception:
        image = _decode_with_opencv(image_bytes)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (img_height, img_width))

    # Width becomes the time dimension (matches training notebook)
    image = tf.transpose(image, perm=[1, 0, 2])

    return image
