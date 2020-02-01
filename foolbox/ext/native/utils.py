from typing import Optional, Tuple
import eagerpy as ep
import foolbox
import warnings

from .types import Bounds
from .models import Model


def accuracy(fmodel: Model, inputs, labels) -> float:
    inputs_, labels_ = ep.astensors(inputs, labels)
    del inputs, labels

    logits = fmodel(inputs_)
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels_).float32().mean()
    return accuracy.item()


def samples(
    fmodel,
    dataset="imagenet",
    index=0,
    batchsize=1,
    shape: Tuple[int, int] = (224, 224),
    data_format: Optional[str] = None,
    bounds: Optional[Bounds] = None,
):
    if hasattr(fmodel, "data_format"):
        if data_format is None:
            data_format = fmodel.data_format
        elif data_format != fmodel.data_format:
            raise ValueError(
                f"data_format ({data_format}) does not match model.data_format ({fmodel.data_format})"
            )
    elif data_format is None:
        raise ValueError(
            "data_format could not be inferred, please specify it explicitly"
        )

    if bounds is None:
        bounds = fmodel.bounds

    images, labels = foolbox.utils.samples(
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        shape=shape,
        data_format=data_format,
        bounds=bounds,
    )

    if hasattr(fmodel, "dummy") and fmodel.dummy is not None:
        images = ep.from_numpy(fmodel.dummy, images).raw
        labels = ep.from_numpy(fmodel.dummy, labels).raw
    else:
        warnings.warn(f"unknown model type {type(fmodel)}, returning NumPy arrays")

    return images, labels
