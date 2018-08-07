from os.path import join, dirname

with open(join(dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from . import models  # noqa: F401
from . import criteria  # noqa: F401
from . import distances  # noqa: F401
from . import attacks  # noqa: F401
from . import utils  # noqa: F401
from . import gradient_estimators  # noqa: F401

from .adversarial import Adversarial  # noqa: F401
