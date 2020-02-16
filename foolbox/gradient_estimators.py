from typing import Callable, Tuple, Type
import eagerpy as ep
from .types import BoundsInput, Bounds
from .attacks.base import Attack


def evolutionary_strategies_gradient_estimator(
    AttackCls: Type[Attack],
    *,
    samples: int,
    sigma: float,
    bounds: BoundsInput,
    clip: bool,
) -> Type[Attack]:

    if not hasattr(AttackCls, "value_and_grad"):
        raise ValueError(
            "This attack does not support gradient estimators."
        )  # pragma: no cover

    bounds = Bounds(*bounds)

    class GradientEstimator(AttackCls):  # type: ignore
        def value_and_grad(
            self, loss_fn: Callable[[ep.Tensor], ep.Tensor], x: ep.Tensor,
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            value = loss_fn(x)

            gradient = ep.zeros_like(x)
            for k in range(samples // 2):
                noise = ep.normal(x, shape=x.shape)

                pos_theta = x + sigma * noise
                neg_theta = x - sigma * noise

                if clip:
                    pos_theta = pos_theta.clip(*bounds)
                    neg_theta = neg_theta.clip(*bounds)

                pos_loss = loss_fn(pos_theta)
                neg_loss = loss_fn(neg_theta)

                gradient += (pos_loss - neg_loss) * noise

            gradient /= 2 * sigma * 2 * samples

            return value, gradient

    GradientEstimator.__name__ = AttackCls.__name__ + "WithESGradientEstimator"
    GradientEstimator.__qualname__ = AttackCls.__qualname__ + "WithESGradientEstimator"
    return GradientEstimator


es_gradient_estimator = evolutionary_strategies_gradient_estimator
