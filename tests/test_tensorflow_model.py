import pytest
import numpy as np
import tensorflow as tf

from foolbox.ext.native.models import TensorFlowModel


@pytest.mark.parametrize("data_format", ["channels_first", "channels_last"])
@pytest.mark.parametrize("model_api", ["sequential", "subclassing", "functional"])
def test_tensorflow_model(data_format, model_api):
    tf.keras.backend.set_image_data_format(data_format)

    channels = num_classes = 10
    batch_size = 8
    h = w = 32
    bounds = (0, 1)

    if model_api == "sequential":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    elif model_api == "subclassing":

        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.pool = tf.keras.layers.GlobalAveragePooling2D()

            def call(self, x):
                x = self.pool(x)
                return x

        model = Model()
    elif model_api == "functional":
        if tf.keras.backend.image_data_format() == "channels_first":
            x = x_ = tf.keras.Input(shape=(channels, h, w))
        else:
            x = x_ = tf.keras.Input(shape=(h, w, channels))
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = tf.keras.Model(inputs=x_, outputs=x)

    fmodel = TensorFlowModel(model, bounds=bounds)

    np.random.seed(0)
    if tf.keras.backend.image_data_format() == "channels_first":
        x = np.random.uniform(*bounds, size=(batch_size, channels, h, w)).astype(
            np.float32
        )
    else:
        x = np.random.uniform(*bounds, size=(batch_size, h, w, channels)).astype(
            np.float32
        )
    x = tf.constant(x)
    y = np.arange(batch_size) % num_classes
    y = tf.constant(y)

    output = fmodel.forward(x)
    assert output.shape == (batch_size, num_classes)
    assert isinstance(output, tf.Tensor)

    output = fmodel.gradient(x, y)
    assert output.shape == x.shape
    assert isinstance(output, tf.Tensor)
