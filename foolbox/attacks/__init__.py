# flake8: noqa

from .base import Attack
from .gradient import GradientAttack, GradientSignAttack, FGSM
from .iterative_gradient import IterativeGradientAttack, IterativeGradientSignAttack
from .deepfool import DeepFoolAttack, DeepFoolL2Attack, DeepFoolLinfinityAttack
from .saliency import SaliencyMapAttack
from .blur import GaussianBlurAttack
from .contrast import ContrastReductionAttack
from .localsearch import SinglePixelAttack, LocalSearchAttack
from .additive_noise import (
    AdditiveNoiseAttack,
    AdditiveUniformNoiseAttack,
    AdditiveGaussianNoiseAttack,
)
from .blended_noise import BlendedUniformNoiseAttack
from .saltandpepper import SaltAndPepperNoiseAttack
from .precomputed import PrecomputedAdversarialsAttack
from .boundary_attack import BoundaryAttack
from .pointwise import PointwiseAttack
from .newtonfool import NewtonFoolAttack
from .adef_attack import ADefAttack
from .spatial import SpatialAttack
from .carlini_wagner import CarliniWagnerL2Attack
from .ead import EADAttack
from .decoupled_direction_norm import DecoupledDirectionNormL2Attack
from .inversion import InversionAttack
from .virtual_adversarial import VirtualAdversarialAttack
from .inversion import InversionAttack
from .binarization import BinarizationRefinementAttack
from .gen import GenAttack
from .hop_skip_jump_attack import HopSkipJumpAttack, BoundaryAttackPlusPlus
from .iterative_projected_gradient import (
    LinfinityBasicIterativeAttack,
    BasicIterativeMethod,
    BIM,
)
from .iterative_projected_gradient import L1BasicIterativeAttack
from .iterative_projected_gradient import L2BasicIterativeAttack

from .iterative_projected_gradient import (
    ProjectedGradientDescentAttack,
    ProjectedGradientDescent,
    PGD,
)
from .iterative_projected_gradient import (
    RandomStartProjectedGradientDescentAttack,
    RandomProjectedGradientDescent,
    RandomPGD,
)
from .iterative_projected_gradient import (
    MomentumIterativeAttack,
    MomentumIterativeMethod,
)
from .iterative_projected_gradient import SparseL1BasicIterativeAttack
from .iterative_projected_gradient import AdamL1BasicIterativeAttack
from .iterative_projected_gradient import AdamL2BasicIterativeAttack
from .iterative_projected_gradient import (
    AdamProjectedGradientDescentAttack,
    AdamProjectedGradientDescent,
    AdamPGD,
)
from .iterative_projected_gradient import (
    AdamRandomStartProjectedGradientDescentAttack,
    AdamRandomProjectedGradientDescent,
    AdamRandomPGD,
)
