import functools
import pytest
import eagerpy as ep

import foolbox
import foolbox.ext.native as fbn


models = {}


def pytest_addoption(parser):
    parser.addoption("--backend")


@pytest.fixture(scope="session")
def dummy(request):
    backend = request.config.option.backend
    if backend is None:
        pytest.skip()
    return ep.utils.get_dummy(backend)


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


def pytorch_simple_model(device):
    import torch

    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    bounds = (0, 1)
    fmodel = fbn.PyTorchModel(
        model, bounds=bounds, device=device, preprocessing=dict(flip_axis=-3)
    )

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("pytorch")
def pytorch_simple_model_string():
    return pytorch_simple_model("cpu")


@register("pytorch")
def pytorch_simple_model_object():
    import torch

    return pytorch_simple_model(torch.device("cpu"))


@register("pytorch")
def pytorch_resnet18():
    import torch
    import torchvision.models as models

    if torch.cuda.is_available():
        pytest.skip()

    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fbn.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_sequential_cpu():
    import tensorflow as tf

    with tf.device("cpu"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds, device="cpu")

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
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

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
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

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_resnet50():
    import tensorflow as tf

    if not tf.test.is_gpu_available():
        pytest.skip()

    model = tf.keras.applications.ResNet50(weights="imagenet")
    preprocessing = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel = fbn.TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
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

    x, _ = fbn.samples(
        fmodel, dataset="imagenet", batchsize=16, data_format="channels_last"
    )
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


def foolbox2_simple_model(channel_axis):
    class Foolbox2DummyModel(foolbox.models.base.Model):
        def __init__(self):
            super().__init__(
                bounds=(0, 1), channel_axis=channel_axis, preprocessing=(0, 1)
            )

        def forward(self, inputs):
            if channel_axis == 1:
                return inputs.mean(axis=(2, 3))
            elif channel_axis == 3:
                return inputs.mean(axis=(1, 2))

        def num_classes(self):
            return 3

    model = Foolbox2DummyModel()
    fmodel = fbn.Foolbox2Model(model)

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(fmodel.forward(x.tensor)).argmax(axis=-1)
    return fmodel, x, y


@register("numpy")
def foolbox2_simple_model_1():
    return foolbox2_simple_model(1)


@register("numpy")
def foolbox2_simple_model_3():
    return foolbox2_simple_model(3)


@pytest.fixture(scope="session", params=list(models.keys()))
def fmodel_and_data(request):
    return models[request.param](request)
