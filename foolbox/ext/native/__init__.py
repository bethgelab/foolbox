from os.path import join, dirname

with open(join(dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()


from . import models  # noqa: F401
from . import attacks  # noqa: F401
