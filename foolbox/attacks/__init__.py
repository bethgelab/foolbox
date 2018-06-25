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
