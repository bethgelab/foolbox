from typing import Optional, Callable, Tuple, Dict, Any, List, NamedTuple
import functools
import pytest
import eagerpy as ep

import foolbox
import foolbox as fbn

ModelAndData = Tuple[fbn.Model, ep.Tensor, ep.Tensor]
CallableModelAndDataAndDescription = NamedTuple(
    "CallableModelAndDataAndDescription",
    [
        ("model_fn", Callable[..., ModelAndData]),
        ("real", bool),
        ("low_dimensional_input", bool),
    ],
)
ModeAndDataAndDescription = NamedTuple(
    "ModeAndDataAndDescription",
    [("model_and_data", ModelAndData), ("real", bool), ("low_dimensional_input", bool)],
)

models: Dict[str, CallableModelAndDataAndDescription] = {}
models_for_attacks: List[str] = []


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--backend")
    parser.addoption("--skipslow", action="store_true")


@pytest.fixture(scope="session")
def dummy(request: Any) -> ep.Tensor:
    backend: Optional[str] = request.config.option.backend
    if backend is None or backend == "none":
        pytest.skip()
        assert False
    return ep.utils.get_dummy(backend)


def register(
    backend: str,
    *,
    real: bool = False,
    low_dimensional_input: bool = False,
    attack: bool = True,
) -> Callable[[Callable], Callable]:
    def decorator(f: Callable[[Any], ModelAndData]) -> Callable[[Any], ModelAndData]:
        @functools.wraps(f)
        def model(request: Any) -> ModelAndData:
            if request.config.option.backend != backend:
                pytest.skip()
            return f(request)

        global models
        global real_models

        models[model.__name__] = CallableModelAndDataAndDescription(
            model_fn=model, real=real, low_dimensional_input=low_dimensional_input
        )
        if attack:
            models_for_attacks.append(model.__name__)
        return model

    return decorator


def pytorch_simple_model(
    device: Any = None, preprocessing: fbn.types.Preprocessing = None
) -> ModelAndData:
    import torch

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    bounds = (0, 1)
    fmodel = fbn.PyTorchModel(
        model, bounds=bounds, device=device, preprocessing=preprocessing
    )

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=8)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("pytorch", low_dimensional_input=True)
def pytorch_simple_model_default(request: Any) -> ModelAndData:
    return pytorch_simple_model()


@register("pytorch", low_dimensional_input=True)
def pytorch_simple_model_default_flip(request: Any) -> ModelAndData:
    return pytorch_simple_model(preprocessing=dict(flip_axis=-3))


@register("pytorch", attack=False, low_dimensional_input=True)
def pytorch_simple_model_default_cpu_native_tensor(request: Any) -> ModelAndData:
    import torch

    mean = 0.05 * torch.arange(3).float()
    std = torch.ones(3) * 2
    return pytorch_simple_model("cpu", preprocessing=dict(mean=mean, std=std, axis=-3))


@register("pytorch", attack=False, low_dimensional_input=True)
def pytorch_simple_model_default_cpu_eagerpy_tensor(request: Any) -> ModelAndData:
    mean = 0.05 * ep.torch.arange(3).float32()
    std = ep.torch.ones(3) * 2
    return pytorch_simple_model("cpu", preprocessing=dict(mean=mean, std=std, axis=-3))


@register("pytorch", attack=False, low_dimensional_input=True)
def pytorch_simple_model_string(request: Any) -> ModelAndData:
    return pytorch_simple_model("cpu")


@register("pytorch", attack=False, low_dimensional_input=True)
def pytorch_simple_model_object(request: Any) -> ModelAndData:
    import torch

    return pytorch_simple_model(torch.device("cpu"))


@register("pytorch", real=True, low_dimensional_input=True)
def pytorch_mnist(request: Any) -> ModelAndData:
    fmodel = fbn.zoo.ModelLoader.get().load(
        "examples/zoo/mnist/", module_name="foolbox_model"
    )
    x, y = fbn.samples(fmodel, dataset="mnist", batchsize=8)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("pytorch", real=True)
def pytorch_shufflenetv2(request: Any) -> ModelAndData:
    if request.config.option.skipslow:
        pytest.skip()

    import torchvision.models as models

    model = models.shufflenet_v2_x0_5(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fbn.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=8)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


def tensorflow_simple_sequential(
    device: Optional[str] = None, preprocessing: fbn.types.Preprocessing = None
) -> ModelAndData:
    import tensorflow as tf

    with tf.device(device):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(
        model, bounds=bounds, device=device, preprocessing=preprocessing
    )

    x, _ = fbn.samples(fmodel, dataset="cifar10", batchsize=8)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow", low_dimensional_input=True)
def tensorflow_simple_sequential_cpu(request: Any) -> ModelAndData:
    return tensorflow_simple_sequential("cpu", None)


@register("tensorflow", low_dimensional_input=True)
def tensorflow_simple_sequential_native_tensors(request: Any) -> ModelAndData:
    import tensorflow as tf

    mean = tf.zeros(1)
    std = tf.ones(1) * 255.0
    return tensorflow_simple_sequential("cpu", dict(mean=mean, std=std))


@register("tensorflow", low_dimensional_input=True)
def tensorflow_simple_sequential_eagerpy_tensors(request: Any) -> ModelAndData:
    mean = ep.tensorflow.zeros(1)
    std = ep.tensorflow.ones(1) * 255.0
    return tensorflow_simple_sequential("cpu", dict(mean=mean, std=std))


@register("tensorflow", low_dimensional_input=True)
def tensorflow_simple_subclassing(request: Any) -> ModelAndData:
    import tensorflow as tf

    class Model(tf.keras.Model):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.pool = tf.keras.layers.GlobalAveragePooling2D()

        def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore
            x = self.pool(x)
            return x

    model = Model()
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds)

    x, _ = fbn.samples(fmodel, dataset="cifar10", batchsize=8)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_functional(request: Any) -> ModelAndData:
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

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=8)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow", real=True)
def tensorflow_mobilenetv3(request: Any) -> ModelAndData:
    if request.config.option.skipslow:
        pytest.skip()

    import tensorflow as tf

    model = tf.keras.applications.MobileNetV3Small(
        weights="imagenet", minimalistic=True
    )
    fmodel = fbn.TensorFlowModel(model, bounds=(0, 255), preprocessing=None,)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=8)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("tensorflow", real=True)
def tensorflow_resnet50(request: Any) -> ModelAndData:
    if request.config.option.skipslow:
        pytest.skip()

    import tensorflow as tf

    if not tf.test.is_gpu_available():
        pytest.skip("ResNet50 test too slow without GPU")

    model = tf.keras.applications.ResNet50(weights="imagenet")
    preprocessing = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel = fbn.TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=8)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("jax", low_dimensional_input=True)
def jax_simple_model(request: Any) -> ModelAndData:
    import jax

    def model(x: Any) -> Any:
        return jax.numpy.mean(x, axis=(1, 2))

    bounds = (0, 1)
    fmodel = fbn.JAXModel(model, bounds=bounds)

    x, _ = fbn.samples(
        fmodel, dataset="cifar10", batchsize=8, data_format="channels_last"
    )
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("numpy")
def numpy_simple_model(request: Any) -> ModelAndData:
    class Model:
        def __call__(self, inputs: Any) -> Any:
            return inputs.mean(axis=(2, 3))

    model = Model()
    with pytest.raises(ValueError):
        fbn.NumPyModel(model, bounds=(0, 1), data_format="foo")

    fmodel = fbn.NumPyModel(model, bounds=(0, 1))
    with pytest.raises(ValueError, match="data_format"):
        x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=8)

    fmodel = fbn.NumPyModel(model, bounds=(0, 1), data_format="channels_first")
    with pytest.warns(UserWarning, match="returning NumPy arrays"):
        x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=8)

    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@pytest.fixture(scope="session", params=list(models.keys()))
def fmodel_and_data_ext(request: Any) -> ModeAndDataAndDescription:
    global models
    model_description = models[request.param]
    model_and_data = model_description.model_fn(request)
    return ModeAndDataAndDescription(model_and_data, *model_description[1:])


@pytest.fixture(scope="session", params=models_for_attacks)
def fmodel_and_data_ext_for_attacks(request: Any) -> ModeAndDataAndDescription:
    global models
    model_description = models[request.param]
    model_and_data = model_description.model_fn(request)
    return ModeAndDataAndDescription(model_and_data, *model_description[1:])


@pytest.fixture(scope="session")
def fmodel_and_data(fmodel_and_data_ext: ModeAndDataAndDescription) -> ModelAndData:
    return fmodel_and_data_ext.model_and_data
