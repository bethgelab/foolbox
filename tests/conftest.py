import functools
import pytest
import eagerpy as ep

import foolbox.ext.native as fbn


models = {}


def pytest_addoption(parser):
    parser.addoption("--backend")


def register(backend):
    def decorator(f):
        @functools.wraps(f)
        def model(request):
            if request.config.option.backend != backend:
                pytest.skip()
            return f()

        models[model.__name__] = model
        return model

    return decorator


@register("pytorch")
def pytorch_simple_model():
    import torch

    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    bounds = (0, 1)
    fmodel = fbn.PyTorchModel(model, bounds=bounds)

    x, _ = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("pytorch")
def pytorch_resnet18():
    import torchvision.models as models

    model = models.resnet18(pretrained=True).cuda().eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fbn.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    x, y = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_sequential():
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds)

    x, _ = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_subclassing():
    import tensorflow as tf

    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.pool = tf.keras.layers.GlobalAveragePooling2D()

        def call(self, x):
            x = self.pool(x)
            return x

    model = Model()
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds)

    x, _ = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_functional():
    import tensorflow as tf

    channels = 3
    h = w = 224
    data_format = tf.keras.backend.image_data_format()
    shape = (channels, h, w) if data_format == "channels_first" else (h, w, channels)
    x = x_ = tf.keras.Input(shape=shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=x_, outputs=x)
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds)

    x, _ = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_resnet50():
    import tensorflow as tf

    model = tf.keras.applications.ResNet50(weights="imagenet")
    preprocessing = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel = fbn.TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)

    x, y = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("jax")
def jax_simple_model():
    import jax

    def model(x):
        return jax.numpy.mean(x, axis=(1, 2))

    bounds = (0, 1)
    fmodel = fbn.JAXModel(model, bounds=bounds)

    x, _ = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@pytest.fixture(scope="session", params=list(models.keys()))
def fmodel_and_data(request):
    return models[request.param](request)
