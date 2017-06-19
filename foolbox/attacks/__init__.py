# flake8: noqa

from .base import Attack
from .gradientsign import FGSM, GradientSignAttack, IterativeGradientSignAttack
from .gradient import GradientAttack, IterativeGradientAttack
from .lbfgs import LBFGSAttack, ApproximateLBFGSAttack
from .deepfool import DeepFoolAttack
from .saliency import SaliencyMapAttack
from .blur import GaussianBlurAttack
from .contrast import ContrastReductionAttack
from .localsearch import SinglePixelAttack, LocalSearchAttack
from .slsqp import SLSQPAttack
from .additive_noise import AdditiveNoiseAttack, AdditiveUniformNoiseAttack, AdditiveGaussianNoiseAttack
from .saltandpepper import SaltAndPepperNoiseAttack
from .precomputed import PrecomputedImagesAttack
