import numpy as np

from quovadis_repro.shap_bpt import output_to_map, sample_to_uint8_rgb


def test_shap_bpt_tree_image_from_grayscale():
    image = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4, 1)
    result = sample_to_uint8_rgb(image, channel_first=False)
    assert result.shape == (4, 4, 3)
    assert result.dtype == np.uint8


def test_shap_bpt_output_single_class():
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    result = output_to_map(values)
    assert result.shape == (4, 4)
