import eagerpy as ep
import foolbox
import warnings

from .devutils import wrap_


def accuracy(fmodel, inputs, labels):
    inputs, labels = wrap_(inputs, labels)
    logits = fmodel.forward(inputs)
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).float32().mean()
    return accuracy.item()


def samples(
    model,
    dataset="imagenet",
    index=0,
    batchsize=1,
    shape=(224, 224),
    data_format=None,
    bounds=None,
):
    if hasattr(model, "data_format"):
        if data_format is None:
            data_format = model.data_format
        elif data_format != model.data_format:
            raise ValueError(
                f"data_format ({data_format}) does not match model.data_format ({model.data_format})"
            )
    elif data_format is None:
        raise ValueError(
            "data_format could not be inferred, please specify it explicitly"
        )

    if bounds is None:
        bounds = model.bounds()

    images, labels = foolbox.utils.samples(
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        shape=shape,
        data_format=data_format,
        bounds=bounds,
    )

    if hasattr(model, "dummy"):
        images = ep.from_numpy(model.dummy, images).tensor
        labels = ep.from_numpy(model.dummy, labels).tensor
    else:
        warnings.warn(f"unknown model type {type(model)}, returning NumPy arrays")

    return images, labels
