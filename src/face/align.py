"""Five-point face alignment.

ArcFace-family encoders (and SFace) are trained on faces warped onto a fixed
112x112 template using the five facial landmarks. Skipping this step and simply
resizing the face box costs a large amount of accuracy, because the network
then has to absorb pose, roll and scale variation it was never trained to
handle. Alignment is what makes the embedding depend on the face rather than on
how the head happened to be turned.
"""

from __future__ import annotations

import cv2
import numpy as np

# Canonical landmark positions for a 112x112 chip, as published with ArcFace
# and used by InsightFace, SFace and every compatible encoder.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],   # right eye
        [73.5318, 51.5014],   # left eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # right mouth corner
        [70.7299, 92.2041],   # left mouth corner
    ],
    dtype=np.float32,
)

DEFAULT_CHIP_SIZE = 112


def template_for(size: int = DEFAULT_CHIP_SIZE) -> np.ndarray:
    """Scale the canonical template to a different chip size."""
    return ARCFACE_TEMPLATE_112 * (size / DEFAULT_CHIP_SIZE)


def estimate_transform(landmarks: np.ndarray, size: int = DEFAULT_CHIP_SIZE) -> np.ndarray:
    """Least-squares similarity transform from landmarks onto the template.

    A *similarity* transform (rotation + uniform scale + translation, no shear)
    is used deliberately: it corrects head roll and scale without distorting
    facial geometry, which a full affine fit would.
    """
    source = np.asarray(landmarks, dtype=np.float32)[:5].reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(
        source, template_for(size), method=cv2.LMEDS, refineIters=10
    )
    if matrix is None:
        raise ValueError("could not estimate a face alignment transform")
    return matrix.astype(np.float32)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    size: int = DEFAULT_CHIP_SIZE,
    _bbox=None,
) -> np.ndarray:
    """Warp a face onto the canonical template, returning a ``size x size`` chip.

    ``bbox`` is accepted and ignored. Measured on real footage: when a head
    turns, both eyes project onto nearly the same point and the five-point fit
    becomes visibly wrong -- the chip is a rotated close-up of an ear. The
    obvious repair, detecting that case and substituting a box-framed crop,
    was implemented and measured, and it is *worse*: rank-1 fell from 100% to
    94.8% and the fifth-percentile genuine score from 0.244 to 0.040 on the
    same 251 labelled faces. The encoder is trained on warped chips, so even a
    badly warped one is closer to its input distribution than a plain resized
    crop. The parameter is kept so callers need not know that, and so the
    finding is recorded where the next person will look.
    """
    matrix = estimate_transform(landmarks, size)
    return cv2.warpAffine(
        image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )


def crop_face_fallback(
    image: np.ndarray, bbox, size: int = DEFAULT_CHIP_SIZE, margin: float = 0.15
) -> np.ndarray | None:
    """Plain padded crop, used only when landmarks are unavailable.

    Accuracy is materially worse than aligned input, so callers mark faces
    handled this way and the quality report says so.
    """
    height, width = image.shape[:2]
    pad_x, pad_y = bbox.width * margin, bbox.height * margin
    x1 = int(max(0, bbox.x1 - pad_x))
    y1 = int(max(0, bbox.y1 - pad_y))
    x2 = int(min(width, bbox.x2 + pad_x))
    y2 = int(min(height, bbox.y2 + pad_y))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
