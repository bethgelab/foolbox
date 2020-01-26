from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()

# internal modules
from . import devutils  # noqa: F401
from . import tensorboard  # noqa: F401

# user-facing modules
from .models import PyTorchModel  # noqa: F401
from .models import TensorFlowModel  # noqa: F401
from .models import JAXModel  # noqa: F401
from .models import Foolbox2Model  # noqa: F401

from .distances import l0  # noqa: F401
from .distances import l1  # noqa: F401
from .distances import l2  # noqa: F401
from .distances import linf  # noqa: F401

from .criteria import Misclassification  # noqa: F401
from .criteria import misclassification  # noqa: F401
from .criteria import TargetedMisclassification  # noqa: F401

from .utils import accuracy  # noqa: F401
from .utils import samples  # noqa: F401

from . import attacks  # noqa: F401

from . import plot  # noqa: F401

from .evaluate import evaluate_l2  # noqa: F401
