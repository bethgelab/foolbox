from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()


from . import models  # noqa: F401
from . import attacks  # noqa: F401
