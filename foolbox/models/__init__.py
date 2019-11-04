"""
Provides classes to wrap existing models in different framworks so
that they provide a unified API to the attacks.

"""

from .base import Model  # noqa: F401
from .base import DifferentiableModel  # noqa: F401

from .wrappers import ModelWrapper  # noqa: F401
from .wrappers import DifferentiableModelWrapper  # noqa: F401
from .wrappers import ModelWithoutGradients  # noqa: F401
from .wrappers import ModelWithEstimatedGradients  # noqa: F401
from .wrappers import CompositeModel  # noqa: F401
from .wrappers import EnsembleAveragedModel  # noqa: F401

from .tensorflow import TensorFlowModel  # noqa: F401
from .tensorflow_eager import TensorFlowEagerModel  # noqa: F401
from .pytorch import PyTorchModel  # noqa: F401
from .keras import KerasModel  # noqa: F401
from .theano import TheanoModel  # noqa: F401
from .lasagne import LasagneModel  # noqa: F401
from .mxnet import MXNetModel  # noqa: F401
from .mxnet_gluon import MXNetGluonModel  # noqa: F401
from .caffe import CaffeModel  # noqa: F401
from .jax import JAXModel  # noqa: F401
