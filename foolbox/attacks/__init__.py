# flake8: noqa

from .base import Attack
from .gradient import GradientAttack, GradientSignAttack, FGSM
from .iterative_gradient import IterativeGradientAttack, IterativeGradientSignAttack
from .lbfgs import LBFGSAttack, ApproximateLBFGSAttack
from .deepfool import DeepFoolAttack, DeepFoolL2Attack, DeepFoolLinfinityAttack
from .saliency import SaliencyMapAttack
from .blur import GaussianBlurAttack
from .contrast import ContrastReductionAttack
from .localsearch import SinglePixelAttack, LocalSearchAttack
from .slsqp import SLSQPAttack
from .additive_noise import AdditiveNoiseAttack, AdditiveUniformNoiseAttack, AdditiveGaussianNoiseAttack
from .blended_noise import BlendedUniformNoiseAttack
from .saltandpepper import SaltAndPepperNoiseAttack
from .precomputed import PrecomputedImagesAttack
from .boundary_attack import BoundaryAttack
from .pointwise import PointwiseAttack
from .binarization import BinarizationRefinementAttack
from .newtonfool import NewtonFoolAttack
from .adef_attack import ADefAttack
from .spatial import SpatialAttack
from .carlini_wagner import CarliniWagnerL2Attack
from .decoupled_direction_norm import DecoupledDirectionNormL2Attack
from .bapp import BoundaryAttackPlusPlus

from .iterative_projected_gradient import LinfinityBasicIterativeAttack, BasicIterativeMethod, BIM
from .iterative_projected_gradient import L1BasicIterativeAttack
from .iterative_projected_gradient import L2BasicIterativeAttack
from .iterative_projected_gradient import ProjectedGradientDescentAttack, ProjectedGradientDescent, PGD
from .iterative_projected_gradient import RandomStartProjectedGradientDescentAttack, RandomProjectedGradientDescent, RandomPGD
from .iterative_projected_gradient import MomentumIterativeAttack, MomentumIterativeMethod
