import pytest
import eagerpy as ep

import foolbox.ext.native as fbn


def test_bounds(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    min_, max_ = fmodel.bounds()
    assert min_ < max_
    assert (x >= min_).all()
    assert (x <= max_).all()


def test_forward_unwrapped(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    logits = ep.astensor(fmodel.forward(x.tensor))
    assert logits.ndim == 2
    assert len(logits) == len(x) == len(y)
    _, num_classes = logits.shape
    assert (y >= 0).all()
    assert (y < num_classes).all()


def test_forward_wrapped(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    assert ep.istensor(x)
    logits = fmodel.forward(x)
    assert ep.istensor(logits)
    assert logits.ndim == 2
    assert len(logits) == len(x) == len(y)
    _, num_classes = logits.shape
    assert (y >= 0).all()
    assert (y < num_classes).all()


def test_pytorch_training_warning(request):
    backend = request.config.option.backend
    if backend != "pytorch":
        pytest.skip()

    import torch

    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    model = Model().train()
    bounds = (0, 1)
    with pytest.warns(UserWarning):
        fbn.PyTorchModel(model, bounds=bounds, device="cpu")
