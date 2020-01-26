import foolbox.ext.native as fbn
import pytest


def test_accracy(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    accuracy = fbn.accuracy(fmodel, x, y)
    assert 0 <= accuracy <= 1
    assert accuracy > 0.5
    y = fmodel.forward(x).argmax(axis=-1)
    accuracy = fbn.accuracy(fmodel, x, y)
    assert accuracy == 1


@pytest.mark.parametrize("batchsize", [1, 8])
def test_samples(fmodel_and_data, batchsize):
    fmodel, x, y = fmodel_and_data
    if hasattr(fmodel, "data_format"):
        x = fbn.samples(fmodel, batchsize=batchsize)
        assert len(x) == batchsize
        x = fbn.samples(fmodel, batchsize=batchsize, data_format=fmodel.data_format)
        assert len(x) == batchsize
        with pytest.raises(ValueError):
            data_format = {
                "channels_first": "channels_last",
                "channels_last": "channels_first",
            }[fmodel.data_format]
            fbn.samples(fmodel, batchsize=batchsize, data_format=data_format)
    else:
        x = fbn.samples(fmodel, batchsize=batchsize, data_format="channels_first")
        assert len(x) == batchsize
