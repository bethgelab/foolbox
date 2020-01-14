from .basic_iterative_method import L2BasicIterativeAttack  # noqa: F401
from .basic_iterative_method import LinfinityBasicIterativeAttack  # noqa: F401
from .fast_gradient_method import L2FastGradientAttack  # noqa: F401
from .fast_gradient_method import LinfinityFastGradientAttack  # noqa: F401
from .carlini_wagner import L2CarliniWagnerAttack  # noqa: F401
from .ead import EADAttack  # noqa: F401
from .projected_gradient_descent import ProjectedGradientDescentAttack  # noqa: F401
from .contrast import L2ContrastReductionAttack  # noqa: F401
from .contrast import BinarySearchContrastReductionAttack  # noqa: F401
from .contrast import LinearSearchContrastReductionAttack  # noqa: F401
from .inversion import InversionAttack  # noqa: F401
from .blended_noise import LinearSearchBlendedUniformNoiseAttack  # noqa: F401
from .brendel_bethge import (  # noqa: F401
    L0BrendelBethgeAttack,
    L1BrendelBethgeAttack,
    L2BrendelBethgeAttack,
    LinfinityBrendelBethgeAttack,
)
from .dataset_attack import DatasetAttack  # noqa: F401
from .additive_noise import L2AdditiveGaussianNoiseAttack  # noqa: F401
from .additive_noise import L2AdditiveUniformNoiseAttack  # noqa: F401
from .additive_noise import LinfAdditiveUniformNoiseAttack  # noqa: F401
from .additive_noise import L2RepeatedAdditiveGaussianNoiseAttack  # noqa: F401
from .additive_noise import L2RepeatedAdditiveUniformNoiseAttack  # noqa: F401
from .additive_noise import LinfRepeatedAdditiveUniformNoiseAttack  # noqa: F401
from .saltandpepper import SaltAndPepperNoiseAttack  # noqa: F401
from .binarization import BinarizationRefinementAttack  # noqa: F401
from .boundary_attack import BoundaryAttack  # noqa: F401
from .blur import GaussianBlurAttack  # noqa: F401
from .deepfool import L2DeepFoolAttack, LinfDeepFoolAttack  # noqa: F401

FGM = L2FastGradientAttack
FGSM = LinfinityFastGradientAttack
PGD = ProjectedGradientDescentAttack
