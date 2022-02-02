from typing import Tuple, Any
import pytest
import eagerpy as ep
import numpy as np
import copy

import foolbox as fbn

ModelAndData = Tuple[fbn.Model, ep.Tensor, ep.Tensor]


def test_bounds(fmodel_and_data: ModelAndData) -> None:
    fmodel, x, y = fmodel_and_data
    min_, max_ = fmodel.bounds
    assert min_ < max_
    assert (x >= min_).all()
    assert (x <= max_).all()


def test_forward_unwrapped(fmodel_and_data: ModelAndData) -> None:
    fmodel, x, y = fmodel_and_data
    logits = ep.astensor(fmodel(x.raw))
    assert logits.ndim == 2
    assert len(logits) == len(x) == len(y)
    _, num_classes = logits.shape
    assert (y >= 0).all()
    assert (y < num_classes).all()
    if hasattr(logits.raw, "requires_grad"):
        assert not logits.raw.requires_grad


def test_forward_wrapped(fmodel_and_data: ModelAndData) -> None:
    fmodel, x, y = fmodel_and_data
    assert ep.istensor(x)
    logits = fmodel(x)
    assert ep.istensor(logits)
    assert logits.ndim == 2
    assert len(logits) == len(x) == len(y)
    _, num_classes = logits.shape
    assert (y >= 0).all()
    assert (y < num_classes).all()
    if hasattr(logits.raw, "requires_grad"):
        assert not logits.raw.requires_grad


def test_pytorch_training_warning(request: Any) -> None:
    backend = request.config.option.backend
    if backend != "pytorch":
        pytest.skip()

    import torch

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = Model().train()
    bounds = (0, 1)
    with pytest.warns(UserWarning):
        fbn.PyTorchModel(model, bounds=bounds, device="cpu")


def test_pytorch_invalid_model(request: Any) -> None:
    backend = request.config.option.backend
    if backend != "pytorch":
        pytest.skip()

    class Model:
        def forward(self, x: Any) -> Any:
            return x

    model = Model()
    bounds = (0, 1)
    with pytest.raises(ValueError, match="torch.nn.Module"):
        fbn.PyTorchModel(model, bounds=bounds)


def test_jax_no_data_format(request: Any) -> None:
    backend = request.config.option.backend
    if backend != "jax":
        pytest.skip()

    class Model:
        def __call__(self, x: Any) -> Any:
            return x

    model = Model()
    bounds = (0, 1)
    fmodel = fbn.JAXModel(model, bounds, data_format=None)
    with pytest.raises(AttributeError):
        fmodel.data_format


@pytest.mark.parametrize("bounds", [(0, 1), (-1.0, 1.0), (0, 255), (-32768, 32767)])
def test_transform_bounds(
    fmodel_and_data: ModelAndData, bounds: fbn.types.BoundsInput
) -> None:
    fmodel1, x, y = fmodel_and_data
    logits1 = fmodel1(x)
    min1, max1 = fmodel1.bounds

    fmodel2 = fmodel1.transform_bounds(bounds)
    min2, max2 = fmodel2.bounds
    x2 = (x - min1) / (max1 - min1) * (max2 - min2) + min2
    logits2 = fmodel2(x2)

    np.testing.assert_allclose(logits1.numpy(), logits2.numpy(), rtol=1e-4, atol=1e-4)

    # to make sure fmodel1 is not changed in-place
    logits1b = fmodel1(x)
    np.testing.assert_allclose(logits1.numpy(), logits1b.numpy(), rtol=2e-6)

    fmodel1c = fmodel2.transform_bounds(fmodel1.bounds)
    logits1c = fmodel1c(x)
    np.testing.assert_allclose(logits1.numpy(), logits1c.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bounds", [(0, 1), (-1.0, 1.0), (0, 255), (-32768, 32767)])
def test_transform_bounds_inplace(
    fmodel_and_data: ModelAndData, bounds: fbn.types.BoundsInput
) -> None:
    fmodel, x, y = fmodel_and_data
    fmodel = copy.copy(fmodel)  # to avoid interference with other tests

    if not isinstance(fmodel, fbn.models.base.ModelWithPreprocessing):
        pytest.skip()
        assert False
    logits1 = fmodel(x)
    min1, max1 = fmodel.bounds

    fmodel.transform_bounds(bounds, inplace=True)
    min2, max2 = fmodel.bounds
    x2 = (x - min1) / (max1 - min1) * (max2 - min2) + min2
    logits2 = fmodel(x2)

    np.testing.assert_allclose(logits1.numpy(), logits2.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bounds", [(0, 1), (-1.0, 1.0), (0, 255), (-32768, 32767)])
@pytest.mark.parametrize("manual", [True, False])
def test_transform_bounds_wrapper(
    fmodel_and_data: ModelAndData, bounds: fbn.types.BoundsInput, manual: bool
) -> None:
    fmodel1, x, y = fmodel_and_data
    fmodel1 = copy.copy(fmodel1)  # to avoid interference with other tests

    logits1 = fmodel1(x)
    min1, max1 = fmodel1.bounds

    fmodel2: fbn.Model
    if manual:
        fmodel2 = fbn.models.TransformBoundsWrapper(fmodel1, bounds)
    else:
        if not isinstance(fmodel1, fbn.models.base.ModelWithPreprocessing):
            pytest.skip()
            assert False
        fmodel2 = fmodel1.transform_bounds(bounds, wrapper=True)
        with pytest.raises(ValueError, match="cannot both be True"):
            fmodel1.transform_bounds(bounds, inplace=True, wrapper=True)
    assert isinstance(fmodel2, fbn.models.TransformBoundsWrapper)
    min2, max2 = fmodel2.bounds
    x2 = (x - min1) / (max1 - min1) * (max2 - min2) + min2
    logits2 = fmodel2(x2)

    np.testing.assert_allclose(logits1.numpy(), logits2.numpy(), rtol=1e-4, atol=1e-4)

    # to make sure fmodel1 is not changed in-place
    logits1b = fmodel1(x)
    np.testing.assert_allclose(logits1.numpy(), logits1b.numpy(), rtol=2e-6)

    fmodel1c = fmodel2.transform_bounds(fmodel1.bounds)
    logits1c = fmodel1c(x)
    np.testing.assert_allclose(logits1.numpy(), logits1c.numpy(), rtol=1e-4, atol=1e-4)

    # to make sure fmodel2 is not changed in-place
    logits2b = fmodel2(x2)
    np.testing.assert_allclose(logits2.numpy(), logits2b.numpy(), rtol=2e-6)

    fmodel2.transform_bounds(fmodel1.bounds, inplace=True)
    logits1d = fmodel2(x)
    np.testing.assert_allclose(logits1d.numpy(), logits1.numpy(), rtol=2e-6)


def test_preprocessing(fmodel_and_data: ModelAndData) -> None:
    fmodel, x, y = fmodel_and_data
    if not isinstance(fmodel, fbn.models.base.ModelWithPreprocessing):
        pytest.skip()
        assert False

    preprocessing = dict(mean=[3, 3, 3], std=[5, 5, 5], axis=-3)
    fmodel = fbn.models.base.ModelWithPreprocessing(
        fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
    )

    preprocessing = dict(mean=[3, 3, 3], axis=-3)
    fmodel = fbn.models.base.ModelWithPreprocessing(
        fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
    )

    preprocessing = dict(mean=np.array([3, 3, 3]), axis=-3)
    fmodel = fbn.models.base.ModelWithPreprocessing(
        fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
    )

    # std -> foo
    preprocessing = dict(mean=[3, 3, 3], foo=[5, 5, 5], axis=-3)
    with pytest.raises(ValueError):
        fmodel = fbn.models.base.ModelWithPreprocessing(
            fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
        )

    # axis positive
    preprocessing = dict(mean=[3, 3, 3], std=[5, 5, 5], axis=1)
    with pytest.raises(ValueError):
        fmodel = fbn.models.base.ModelWithPreprocessing(
            fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
        )

    preprocessing = dict(mean=3, std=5)
    fmodel = fbn.models.base.ModelWithPreprocessing(
        fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
    )

    # axis with 1D mean
    preprocessing = dict(mean=3, std=[5, 5, 5], axis=-3)
    with pytest.raises(ValueError):
        fmodel = fbn.models.base.ModelWithPreprocessing(
            fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
        )

    # axis with 1D std
    preprocessing = dict(mean=[3, 3, 3], std=5, axis=-3)
    with pytest.raises(ValueError):
        fmodel = fbn.models.base.ModelWithPreprocessing(
            fmodel._model, fmodel.bounds, fmodel.dummy, preprocessing
        )


def test_transform_bounds_wrapper_data_format() -> None:
    class Model(fbn.models.Model):
        data_format = "channels_first"

        @property
        def bounds(self) -> fbn.types.Bounds:
            return fbn.types.Bounds(0, 1)

        def __call__(self, inputs: fbn.models.base.T) -> fbn.models.base.T:
            return inputs

    model = Model()
    wrapped_model = fbn.models.TransformBoundsWrapper(model, (0, 1))
    assert fbn.attacks.base.get_channel_axis(
        model, 3
    ) == fbn.attacks.base.get_channel_axis(wrapped_model, 3)
    assert hasattr(wrapped_model, "data_format")
    assert not hasattr(wrapped_model, "not_data_format")


def test_transform_bounds_wrapper_missing_data_format() -> None:
    class Model(fbn.models.Model):
        @property
        def bounds(self) -> fbn.types.Bounds:
            return fbn.types.Bounds(0, 1)

        def __call__(self, inputs: fbn.models.base.T) -> fbn.models.base.T:
            return inputs

    model = Model()
    wrapped_model = fbn.models.TransformBoundsWrapper(model, (0, 1))
    assert not hasattr(wrapped_model, "data_format")
    assert not hasattr(wrapped_model, "not_data_format")
