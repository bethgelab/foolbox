import pytest
import numpy as np

from foolbox.models.base import _create_preprocessing_fn

params = [
    (0, 1),
    (0, 255),
    (128, 1),
    (128, 255),
    (0., 1.),
    (0., 255.),
    (128., 1.),
    (128., 255.),
    (np.array([1., 2., 3.], dtype=np.float64),
     np.array([1., 2., 3.], dtype=np.float64)),
]


@pytest.mark.parametrize('params', params)
def test_preprocessing(params, image):
    image_copy = image.copy()
    preprocessing = _create_preprocessing_fn(params)
    preprocessed, backward = preprocessing(image)
    assert image.shape == preprocessed.shape
    assert image.dtype == preprocessed.dtype
    assert np.allclose((image - params[0]) / params[1], preprocessed)
    assert np.all(image == image_copy)
    assert callable(backward)
    dmdp = image
    dmdx = backward(dmdp)
    assert np.all(image == image_copy)
    assert image.shape == dmdx.shape
    assert image.dtype == dmdx.dtype
