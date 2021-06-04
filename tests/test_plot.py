import pytest
import eagerpy as ep
import foolbox as fbn


def test_plot(dummy: ep.Tensor) -> None:
    # just tests that the calls don't throw any errors
    images = ep.zeros(dummy, (10, 3, 32, 32))
    fbn.plot.images(images)
    fbn.plot.images(images, n=3)
    fbn.plot.images(images, n=3, data_format="channels_first")
    fbn.plot.images(images, nrows=4)
    fbn.plot.images(images, ncols=3)
    fbn.plot.images(images, nrows=2, ncols=6)
    fbn.plot.images(images, nrows=2, ncols=4)
    # test for single channel images
    images = ep.zeros(dummy, (10, 32, 32, 1))
    fbn.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 3, 3, 3))
        fbn.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 1, 1, 1))
        fbn.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 32, 32))
        fbn.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 3, 32, 32))
        fbn.plot.images(images, data_format="foo")
