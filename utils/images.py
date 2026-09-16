"""Image decoding shared by the web app and the S3 archiver.

Both used to carry their own copy of load_image, and both failed identically on
the first HEIC upload.
"""
import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener

logger = logging.getLogger(__name__)

# Phones shoot HEIC by default and Pillow has no native HEIF support, so
# without this every iPhone upload raises UnidentifiedImageError. Registering
# the opener puts HEIC on the ordinary Image.open path. Must run before the
# first open call.
register_heif_opener()


def _imread(image_bytes):
    """Decode with imageio. Imported lazily: with HEIF registered this path is
    a rare last resort, and the app should not fail to start without it."""
    import imageio.v3 as iio

    return np.asarray(iio.imread(image_bytes))


def load_image(image_bytes):
    """Decode uploaded bytes into an RGB PIL image.

    Raises ValueError describing what was actually decoded when the bytes
    cannot be turned into a usable image.
    """
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        # Format not recognised at all -- worth a second opinion from imageio.
        logger.info("Pillow could not identify the upload; trying imageio")
    except OSError as err:
        # UnidentifiedImageError subclasses OSError, so this must stay second.
        # Reaching here means Pillow knew the format but could not finish
        # decoding: almost always a truncated or interrupted upload. imageio
        # will not do better with the same incomplete bytes.
        raise ValueError(f"Image data is incomplete or corrupt: {err}") from err

    try:
        array = _imread(image_bytes)
    except Exception as err:
        raise ValueError(f"Unsupported or corrupt image format: {err}") from err

    # imageio can return a frame stack rather than a single frame -- e.g. the
    # (1, 1, 512, 3) a HEIC decode produced before the heif opener existed.
    # Image.fromarray accepts only 2 or 3 dimensions.
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim not in (2, 3) or min(array.shape[:2]) < 2:
        # A degenerate shape means the decode produced garbage. Rendering a
        # one-pixel-tall strip would be worse than saying so plainly.
        raise ValueError(
            "Unsupported or corrupt image format "
            f"(decoded to shape {array.shape}, dtype {array.dtype})"
        )

    return Image.fromarray(array).convert("RGB")
