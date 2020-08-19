from typing import TypeVar, Callable, Optional, Tuple, Any
from abc import ABC, abstractmethod
import copy
import eagerpy as ep

from ..types import Bounds, BoundsInput, Preprocessing
from ..devutils import atleast_kd


T = TypeVar("T")
PreprocessArgs = Tuple[Optional[ep.Tensor], Optional[ep.Tensor], Optional[int]]


class Model(ABC):
    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        ...

    @abstractmethod  # noqa: F811
    def __call__(self, inputs: T) -> T:
        """Passes inputs through the model and returns the model's output"""
        ...

    def transform_bounds(self, bounds: BoundsInput) -> "Model":
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        # subclasses can provide more efficient implementations
        return TransformBoundsWrapper(self, bounds)


class TransformBoundsWrapper(Model):
    def __init__(self, model: Model, bounds: BoundsInput):
        self._model = model
        self._bounds = Bounds(*bounds)

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    def __call__(self, inputs: T) -> T:
        x, restore_type = ep.astensor_(inputs)
        y = self._preprocess(x)
        z = self._model(y)
        return restore_type(z)

    def transform_bounds(self, bounds: BoundsInput, inplace: bool = False) -> Model:
        if inplace:
            self._bounds = Bounds(*bounds)
            return self
        else:
            # using the wrapped model instead of self to avoid
            # unnessary sequences of wrappers
            return TransformBoundsWrapper(self._model, bounds)

    def _preprocess(self, inputs: ep.TensorType) -> ep.TensorType:
        if self.bounds == self._model.bounds:
            return inputs

        # from bounds to (0, 1)
        min_, max_ = self.bounds
        x = (inputs - min_) / (max_ - min_)

        # from (0, 1) to wrapped model bounds
        min_, max_ = self._model.bounds
        return x * (max_ - min_) + min_

    @property
    def data_format(self) -> Any:
        return self._model.data_format  # type: ignore


ModelType = TypeVar("ModelType", bound="ModelWithPreprocessing")


class ModelWithPreprocessing(Model):
    def __init__(  # type: ignore
        self,
        model: Callable[..., ep.types.NativeTensor],
        bounds: BoundsInput,
        dummy: ep.Tensor,
        preprocessing: Preprocessing = None,
    ):
        if not callable(model):
            raise ValueError("expected model to be callable")  # pragma: no cover

        self._model = model
        self._bounds = Bounds(*bounds)
        self._dummy = dummy
        self._preprocess_args = self._process_preprocessing(preprocessing)

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def dummy(self) -> ep.Tensor:
        return self._dummy

    def __call__(self, inputs: T) -> T:
        x, restore_type = ep.astensor_(inputs)
        y = self._preprocess(x)
        z = ep.astensor(self._model(y.raw))
        return restore_type(z)

    def transform_bounds(
        self, bounds: BoundsInput, inplace: bool = False, wrapper: bool = False,
    ) -> Model:
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        # more efficient than the base class implementation because it avoids the additional wrapper

        if wrapper:
            if inplace:
                raise ValueError("inplace and wrapper cannot both be True")
            return super().transform_bounds(bounds)

        if self.bounds == bounds:
            if inplace:
                return self
            else:
                return copy.copy(self)

        a, b = self.bounds
        c, d = bounds
        f = (d - c) / (b - a)

        mean, std, flip_axis = self._preprocess_args

        if mean is None:
            mean = ep.zeros(self._dummy, 1)
        mean = f * (mean - a) + c

        if std is None:
            std = ep.ones(self._dummy, 1)
        std = f * std

        if inplace:
            model = self
        else:
            model = copy.copy(self)
        model._bounds = Bounds(*bounds)
        model._preprocess_args = (mean, std, flip_axis)
        return model

    def _preprocess(self, inputs: ep.Tensor) -> ep.Tensor:
        mean, std, flip_axis = self._preprocess_args
        x = inputs
        if flip_axis is not None:
            x = x.flip(axis=flip_axis)
        if mean is not None:
            x = x - mean
        if std is not None:
            x = x / std
        assert x.dtype == inputs.dtype
        return x

    def _process_preprocessing(self, preprocessing: Preprocessing) -> PreprocessArgs:
        if preprocessing is None:
            preprocessing = dict()

        unsupported = set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"}
        if len(unsupported) > 0:
            raise ValueError(f"unknown preprocessing key: {unsupported.pop()}")

        mean = preprocessing.get("mean", None)
        std = preprocessing.get("std", None)
        axis = preprocessing.get("axis", None)
        flip_axis = preprocessing.get("flip_axis", None)

        def to_tensor(x: Any) -> Optional[ep.Tensor]:
            if x is None:
                return None
            if isinstance(x, ep.Tensor):
                return x
            try:
                y = ep.astensor(x)  # might raise ValueError
                if not isinstance(y, type(self._dummy)):
                    raise ValueError
                return y
            except ValueError:
                return ep.from_numpy(self._dummy, x)

        mean_ = to_tensor(mean)
        std_ = to_tensor(std)

        def apply_axis(x: Optional[ep.Tensor], axis: int) -> Optional[ep.Tensor]:
            if x is None:
                return None
            if x.ndim != 1:
                raise ValueError(f"non-None axis requires a 1D tensor, got {x.ndim}D")
            if axis >= 0:
                raise ValueError(
                    "expected axis to be None or negative, -1 refers to the last axis"
                )
            return atleast_kd(x, -axis)

        if axis is not None:
            mean_ = apply_axis(mean_, axis)
            std_ = apply_axis(std_, axis)

        return mean_, std_, flip_axis
