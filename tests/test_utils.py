from typing import Tuple
import foolbox.ext.native as fbn
import eagerpy as ep
import pytest

ModelAndData = Tuple[fbn.Model, ep.Tensor, ep.Tensor]


def test_accuracy(fmodel_and_data: ModelAndData) -> None:
    fmodel, x, y = fmodel_and_data
    accuracy = fbn.accuracy(fmodel, x, y)
    assert 0 <= accuracy <= 1
    assert accuracy > 0.5
    y = fmodel(x).argmax(axis=-1)
    accuracy = fbn.accuracy(fmodel, x, y)
    assert accuracy == 1


@pytest.mark.parametrize("batchsize", [1, 8])
def test_samples(fmodel_and_data: ModelAndData, batchsize: int) -> None:
    fmodel, _, _ = fmodel_and_data
    if hasattr(fmodel, "data_format"):
        data_format = fmodel.data_format  # type: ignore
        x, y = fbn.samples(fmodel, batchsize=batchsize)
        assert len(x) == len(y) == batchsize
        assert not ep.istensor(x)
        assert not ep.istensor(y)
        x, y = fbn.samples(fmodel, batchsize=batchsize, data_format=data_format)
        assert len(x) == len(y) == batchsize
        assert not ep.istensor(x)
        assert not ep.istensor(y)
        with pytest.raises(ValueError):
            data_format = {
                "channels_first": "channels_last",
                "channels_last": "channels_first",
            }[data_format]
            fbn.samples(fmodel, batchsize=batchsize, data_format=data_format)
    else:
        x, y = fbn.samples(fmodel, batchsize=batchsize, data_format="channels_first")
        assert len(x) == len(y) == batchsize
        assert not ep.istensor(x)
        assert not ep.istensor(y)
        with pytest.raises(ValueError):
            fbn.samples(fmodel, batchsize=batchsize)
