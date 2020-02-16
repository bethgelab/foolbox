from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()

# internal modules
# ----------------

from . import devutils  # noqa: F401
from . import tensorboard  # noqa: F401
from . import types  # noqa: F401

# user-facing modules
# -------------------

from .distances import Distance  # noqa: F401
from . import distances  # noqa: F401

from .criteria import Criterion  # noqa: F401
from .criteria import Misclassification  # noqa: F401
from .criteria import TargetedMisclassification  # noqa: F401

from . import plot  # noqa: F401

from .models import Model  # noqa: F401
from .models import PyTorchModel  # noqa: F401
from .models import TensorFlowModel  # noqa: F401
from .models import JAXModel  # noqa: F401
from .models import NumPyModel  # noqa: F401

from .utils import accuracy  # noqa: F401
from .utils import samples  # noqa: F401

from .attacks import Attack  # noqa: F401
from . import attacks  # noqa: F401

from . import zoo  # noqa: F401

from . import gradient_estimators  # noqa: F401
