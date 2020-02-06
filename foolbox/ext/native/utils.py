from typing import Optional, Tuple, Any
import eagerpy as ep
import foolbox
import warnings

from .types import Bounds
from .models import Model


def accuracy(fmodel: Model, inputs: Any, labels: Any) -> float:
    inputs_, labels_ = ep.astensors(inputs, labels)
    del inputs, labels

    predictions = fmodel(inputs_).argmax(axis=-1)
    accuracy = (predictions == labels_).float32().mean()
    return accuracy.item()


def samples(
    fmodel: Model,
    dataset: str = "imagenet",
    index: int = 0,
    batchsize: int = 1,
    shape: Tuple[int, int] = (224, 224),
    data_format: Optional[str] = None,
    bounds: Optional[Bounds] = None,
) -> Any:
    if hasattr(fmodel, "data_format"):
        if data_format is None:
            data_format = fmodel.data_format  # type: ignore
        elif data_format != fmodel.data_format:  # type: ignore
            raise ValueError(
                f"data_format ({data_format}) does not match model.data_format ({fmodel.data_format})"  # type: ignore
            )
    elif data_format is None:
        raise ValueError(
            "data_format could not be inferred, please specify it explicitly"
        )

    if bounds is None:
        bounds = fmodel.bounds

    images, labels = foolbox.utils.samples(  # type: ignore
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        shape=shape,
        data_format=data_format,
        bounds=bounds,
    )

    if hasattr(fmodel, "dummy") and fmodel.dummy is not None:  # type: ignore
        images = ep.from_numpy(fmodel.dummy, images).raw  # type: ignore
        labels = ep.from_numpy(fmodel.dummy, labels).raw  # type: ignore
    else:
        warnings.warn(f"unknown model type {type(fmodel)}, returning NumPy arrays")

    return images, labels
