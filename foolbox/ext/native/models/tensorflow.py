import eagerpy as ep

from .base import ModelWithPreprocessing


def get_device(device):
    import tensorflow as tf

    if device is None:
        device = tf.device("/GPU:0" if tf.test.is_gpu_available() else "/CPU:0")
    if isinstance(device, str):
        device = tf.device(device)
    return device


class TensorFlowModel(ModelWithPreprocessing):
    def __init__(self, model, bounds, device=None, preprocessing=None):
        import tensorflow as tf

        assert tf.executing_eagerly()
        device = get_device(device)
        with device:
            dummy = ep.tensorflow.zeros(0)
        super().__init__(model, bounds, dummy, preprocessing=preprocessing)

    @property
    def data_format(self):
        import tensorflow as tf

        return tf.keras.backend.image_data_format()
