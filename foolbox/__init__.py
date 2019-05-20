from os.path import join, dirname

with open(join(dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from .rngs import rng  # noqa: F401
from .rngs import nprng  # noqa: F401
from .rngs import set_seeds  # noqa: F401

from . import models  # noqa: F401
from . import criteria  # noqa: F401
from . import distances  # noqa: F401
from . import attacks  # noqa: F401
from . import batch_attacks  # noqa: F401
from . import utils  # noqa: F401
from . import gradient_estimators  # noqa: F401

from .adversarial import Adversarial  # noqa: F401
from .yielding_adversarial import YieldingAdversarial  # noqa: F401

from .batching import run_parallel  # noqa: F401
from .batching import run_sequential  # noqa: F401
