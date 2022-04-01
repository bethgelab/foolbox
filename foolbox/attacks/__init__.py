from .additive_noise import (  # noqa: F401
    L2AdditiveGaussianNoiseAttack, L2AdditiveUniformNoiseAttack,
    L2ClippingAwareAdditiveGaussianNoiseAttack,
    L2ClippingAwareAdditiveUniformNoiseAttack,
    L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack,
    L2ClippingAwareRepeatedAdditiveUniformNoiseAttack,
    L2RepeatedAdditiveGaussianNoiseAttack,
    L2RepeatedAdditiveUniformNoiseAttack, LinfAdditiveUniformNoiseAttack,
    LinfRepeatedAdditiveUniformNoiseAttack)
from .base import Attack  # noqa: F401
from .basic_iterative_method import (L1AdamBasicIterativeAttack,  # noqa: F401
                                     L1BasicIterativeAttack,
                                     L2AdamBasicIterativeAttack,
                                     L2BasicIterativeAttack,
                                     LinfAdamBasicIterativeAttack,
                                     LinfBasicIterativeAttack)
from .binarization import BinarizationRefinementAttack  # noqa: F401
from .blended_noise import LinearSearchBlendedUniformNoiseAttack  # noqa: F401
from .blur import GaussianBlurAttack  # noqa: F401
from .boundary_attack import BoundaryAttack  # noqa: F401
from .brendel_bethge import (L0BrendelBethgeAttack,  # noqa: F401
                             L1BrendelBethgeAttack, L2BrendelBethgeAttack,
                             LinfinityBrendelBethgeAttack)
from .carlini_wagner import L2CarliniWagnerAttack  # noqa: F401
# FixedEpsilonAttack subclasses
from .contrast import L2ContrastReductionAttack  # noqa: F401
from .contrast_min import (BinarySearchContrastReductionAttack,  # noqa: F401
                           LinearSearchContrastReductionAttack)
from .dataset_attack import DatasetAttack  # noqa: F401
from .ddn import DDNAttack  # noqa: F401
from .deepfool import L2DeepFoolAttack, LinfDeepFoolAttack  # noqa: F401
from .ead import EADAttack  # noqa: F401
from .fast_gradient_method import (L1FastGradientAttack,  # noqa: F401
                                   L2FastGradientAttack,
                                   LinfFastGradientAttack)
from .fast_minimum_norm import (L0FMNAttack, L1FMNAttack,  # noqa: F401
                                L2FMNAttack, LInfFMNAttack)
from .gen_attack import GenAttack  # noqa: F401
from .hop_skip_jump import HopSkipJumpAttack  # noqa: F401
# MinimizatonAttack subclasses
from .inversion import InversionAttack  # noqa: F401
from .newtonfool import NewtonFoolAttack  # noqa: F401
from .pointwise import PointwiseAttack  # noqa: F401
from .projected_gradient_descent import (  # noqa: F401
    L1AdamProjectedGradientDescentAttack, L1ProjectedGradientDescentAttack,
    L2PAdamProjectedGradientDescentAttack, L2ProjectedGradientDescentAttack,
    LinfAdamProjectedGradientDescentAttack, LinfProjectedGradientDescentAttack)
from .saltandpepper import SaltAndPepperNoiseAttack  # noqa: F401
from .sparse_l1_descent_attack import SparseL1DescentAttack  # noqa: F401
from .spatial_attack import SpatialAttack  # noqa: F401
from .virtual_adversarial_attack import VirtualAdversarialAttack  # noqa: F401

# from .blended_noise import LinearSearchBlendedUniformNoiseAttack  # noqa: F401
# from .brendel_bethge import (  # noqa: F401
#     L0BrendelBethgeAttack,
#     L1BrendelBethgeAttack,
#     L2BrendelBethgeAttack,
#     LinfinityBrendelBethgeAttack,
# )
# from .additive_noise import L2AdditiveGaussianNoiseAttack  # noqa: F401
# from .additive_noise import L2AdditiveUniformNoiseAttack  # noqa: F401
# from .additive_noise import LinfAdditiveUniformNoiseAttack  # noqa: F401
# from .additive_noise import L2RepeatedAdditiveGaussianNoiseAttack  # noqa: F401
# from .additive_noise import L2RepeatedAdditiveUniformNoiseAttack  # noqa: F401
# from .additive_noise import LinfRepeatedAdditiveUniformNoiseAttack  # noqa: F401
# from .saltandpepper import SaltAndPepperNoiseAttack  # noqa: F401

FGM = L2FastGradientAttack
FGSM = LinfFastGradientAttack
L1PGD = L1ProjectedGradientDescentAttack
L2PGD = L2ProjectedGradientDescentAttack
LinfPGD = LinfProjectedGradientDescentAttack
PGD = LinfPGD

L1AdamPGD = L1ProjectedGradientDescentAttack
L2AdamPGD = L2ProjectedGradientDescentAttack
LinfAdamPGD = LinfProjectedGradientDescentAttack
AdamPGD = LinfAdamPGD
