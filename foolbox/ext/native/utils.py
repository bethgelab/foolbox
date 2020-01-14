import eagerpy as ep
import foolbox
from .models import PyTorchModel
from .models import TensorFlowModel
from .models import JAXModel
from .models import FoolboxModel


def accuracy(fmodel, inputs, labels):
    logits = ep.astensor(fmodel.forward(inputs))
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
    if data_format is None:
        if isinstance(model, PyTorchModel):
            data_format = "channels_first"
        elif isinstance(model, TensorFlowModel):
            import tensorflow as tf

            data_format = tf.keras.backend.image_data_format()
        elif isinstance(model, FoolboxModel):
            channel_axis = model.foolbox_model.channel_axis()
            if channel_axis == 1:
                data_format = "channels_first"
            elif channel_axis == 3:
                data_format = "channels_last"
            else:
                raise ValueError("data_format could not be inferred from the model")
        else:
            raise ValueError("data_format could not be inferred from the model")
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
    if isinstance(model, PyTorchModel):
        import torch

        images = torch.as_tensor(images).to(model.device)
        labels = torch.as_tensor(labels).to(model.device)
    elif isinstance(model, TensorFlowModel):
        import tensorflow as tf

        with model.device:
            images = tf.convert_to_tensor(images)
            labels = tf.convert_to_tensor(labels)
    elif isinstance(model, JAXModel):
        import jax.numpy as np

        images = np.asarray(images)
        labels = np.asarray(labels)
    elif isinstance(model, FoolboxModel):
        images = images
        labels = labels
    else:
        raise ValueError("data_format could not be inferred from the model")
    return images, labels


def flatten(x: ep.Tensor, keep=1) -> ep.Tensor:
    shape = x.shape[:keep] + (-1,)
    return x.reshape(shape)


def atleast_kd(x: ep.Tensor, k) -> ep.Tensor:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)
