import pkg_resources
__version__ = pkg_resources.get_distribution('foolbox')

from . import models  # type: ignore # noqa: F401
from . import criteria  # type: ignore # noqa: F401
from . import distances  # type: ignore # noqa: F401
from . import attacks  # type: ignore # noqa: F401
from . import utils  # type: ignore # noqa: F401

from .adversarial import Adversarial  # noqa: F401
