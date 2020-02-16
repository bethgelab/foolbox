from typing import Union, Any
import eagerpy as ep

from ..models import Model

from ..criteria import Misclassification

from ..distances import l2

from ..devutils import flatten, atleast_kd

from .base import FixedEpsilonAttack
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs


class VirtualAdversarialAttack(FixedEpsilonAttack):
    """Second-order gradient-based attack on the logits. [#Miy15]_
    The attack calculate an untargeted adversarial perturbation by performing a
    approximated second order optimization step on the KL divergence between
    the unperturbed predictions and the predictions for the adversarial
    perturbation. This attack was originally introduced as the
    Virtual Adversarial Training [#Miy15]_ method.

    Args:
        steps : Number of update steps.
        xi : L2 distance between original image and first adversarial proposal.


    References:
        .. [#Miy15] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae,
            Shin Ishii, "Distributional Smoothing with Virtual Adversarial Training",
            https://arxiv.org/abs/1507.00677
    """

    distance = l2

    def __init__(self, steps: int, xi: float = 1e-6):
        self.steps = steps
        self.xi = xi

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, Misclassification):
            classes = criterion_.labels
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            raise ValueError(
                f"expected labels to have shape ({N},), got {classes.shape}"
            )

        bounds = model.bounds

        def loss_fun(delta: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            assert x.shape[0] == logits.shape[0]
            assert delta.shape == x.shape

            x_hat = x + delta
            logits_hat = model(x_hat)
            loss = ep.kl_div_with_logits(logits, logits_hat).sum()

            return loss

        value_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=False)

        clean_logits = model(x)

        # start with random vector as search vector
        d = ep.normal(x, shape=x.shape, mean=0, stddev=1)
        for it in range(self.steps):
            # normalize proposal to be unit vector
            d = d * self.xi / atleast_kd(ep.norms.l2(flatten(d), axis=-1), x.ndim)

            # use gradient of KL divergence as new search vector
            _, grad = value_and_grad(d, clean_logits)
            d = grad

            # rescale search vector
            d = (bounds[1] - bounds[0]) * d

            if ep.any(ep.norms.l2(flatten(d), axis=-1) < 1e-64):
                raise RuntimeError(  # pragma: no cover
                    "Gradient vanished; this can happen if xi is too small."
                )

        final_delta = epsilon / atleast_kd(ep.norms.l2(flatten(d), axis=-1), d.ndim) * d
        x_adv = ep.clip(x + final_delta, *bounds)
        return restore_type(x_adv)
