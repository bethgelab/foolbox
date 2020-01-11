from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()


from . import models  # noqa: F401
from . import attacks  # noqa: F401
from . import utils  # noqa: F401
from . import plot  # noqa: F401
from . import norms  # noqa: F401
from . import tensorboard  # noqa: F401
from .evaluate import evaluate_l2  # noqa: F401
