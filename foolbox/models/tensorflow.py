from typing import cast, Any
import eagerpy as ep

from ..types import BoundsInput, Preprocessing

from .base import ModelWithPreprocessing


def get_device(device: Any) -> Any:
    import tensorflow as tf

    if device is None:
        device = tf.device("/GPU:0" if tf.test.is_gpu_available() else "/CPU:0")
    if isinstance(device, str):
        device = tf.device(device)
    return device


class TensorFlowModel(ModelWithPreprocessing):
    def __init__(
        self,
        model: Any,
        bounds: BoundsInput,
        device: Any = None,
        preprocessing: Preprocessing = None,
    ):
        import tensorflow as tf

        if not tf.executing_eagerly():
            raise ValueError(
                "TensorFlowModel requires TensorFlow Eager Mode"
            )  # pragma: no cover

        device = get_device(device)
        with device:
            dummy = ep.tensorflow.zeros(0)
        super().__init__(model, bounds, dummy, preprocessing=preprocessing)

        self.device = device

    @property
    def data_format(self) -> str:
        import tensorflow as tf

        return cast(str, tf.keras.backend.image_data_format())
