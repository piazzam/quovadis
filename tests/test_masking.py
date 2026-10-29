import numpy as np

from quovadis_repro.masking import apply_coalition_masks, gaussian_blur, replacement_image


def test_black_masking_keeps_and_removes_regions():
    image = np.ones((4, 4, 1), dtype=np.float32)
    partition = np.zeros((4, 4), dtype=np.int64)
    partition[:, 2:] = 1
    coalitions = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    out = apply_coalition_masks(image, coalitions, partition, strategy="black", black_value=0.0)
    assert np.all(out[0, :, :2] == 0.0)
    assert np.all(out[0, :, 2:] == 1.0)
    assert np.all(out[1, :, :2] == 1.0)
    assert np.all(out[1, :, 2:] == 0.0)


def test_gaussian_replacement_shape():
    image = np.arange(25, dtype=np.float32).reshape(5, 5, 1)
    blurred = gaussian_blur(image, (3, 3))
    assert blurred.shape == image.shape
    assert replacement_image(image, "gaussian_blur", (3, 3)).shape == image.shape
