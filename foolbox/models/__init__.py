"""
Provides classes to wrap existing models in different framworks so
that they provide a unified API to the attacks.

"""

from .base import Model  # noqa: F401
from .base import DifferentiableModel  # noqa: F401

from .wrappers import ModelWrapper  # noqa: F401
from .wrappers import GradientLess  # noqa: F401
from .wrappers import CompositeModel  # noqa: F401

from .tensorflow import TensorFlowModel  # noqa: F401
from .pytorch import PyTorchModel  # noqa: F401
from .keras import KerasModel  # noqa: F401
from .theano import TheanoModel  # noqa: F401
from .lasagne import LasagneModel  # noqa: F401
from .mxnet import MXNetModel  # noqa: F401
