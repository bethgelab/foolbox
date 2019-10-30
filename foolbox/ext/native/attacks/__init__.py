from .basic_iterative_method import L2BasicIterativeAttack  # noqa: F401
from .basic_iterative_method import LinfinityBasicIterativeAttack  # noqa: F401

from .fast_gradient_method import L2FastGradientAttack  # noqa: F401

FGM = L2FastGradientAttack
from .fast_gradient_method import LinfinityFastGradientAttack  # noqa: F401

FGSM = LinfinityFastGradientAttack

from .carlini_wagner import L2CarliniWagnerAttack  # noqa: F401
