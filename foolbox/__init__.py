from os.path import join, dirname

with open(join(dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from . import models  # type: ignore # noqa: F401
from . import criteria  # type: ignore # noqa: F401
from . import distances  # type: ignore # noqa: F401
from . import attacks  # type: ignore # noqa: F401
from . import utils  # type: ignore # noqa: F401

from .adversarial import Adversarial  # noqa: F401
