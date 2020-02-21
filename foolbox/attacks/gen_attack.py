from typing import Optional, Any
from typing_extensions import Tuple
import numpy as np
import eagerpy as ep

from ..devutils import atleast_kd

from ..models import Model

from ..criteria import TargetedMisclassification

from ..distances import linf

from .base import FixedEpsilonAttack
from .base import T
from .base import get_channel_axis
from .base import raise_if_kwargs
import math


class GenAttack(FixedEpsilonAttack):
    """A black-box algorithm for L-infinity adversarials. [#Alz18]_

    This attack is performs a genetic search in order to find an adversarial
    perturbation in a black-box scenario in as few queries as possible.

    References:
        .. [#Alz18] Moustafa Alzantot, Yash Sharma, Supriyo Chakraborty, Huan Zhang,
           Cho-Jui Hsieh, Mani Srivastava,
           "GenAttack: Practical Black-box Attacks with Gradient-Free
           Optimization",
           https://arxiv.org/abs/1805.11090

    """

    def __init__(
        self,
        *,
        steps: int = 1000,
        population: int = 10,
        mutation_probability: float = 0.10,
        mutation_range: float = 0.15,
        sampling_temperature: float = 0.3,
        channel_axis: Optional[int] = None,
        reduced_dims: Optional[Tuple[int, int]] = None,
    ):
        self.steps = steps
        self.population = population
        self.min_mutation_probability = mutation_probability
        self.min_mutation_range = mutation_range
        self.sampling_temperature = sampling_temperature
        self.channel_axis = channel_axis
        self.reduced_dims = reduced_dims

    distance = linf

    def apply_noise(
        self, x: ep.TensorType, noise: ep.TensorType, epsilon: float
    ) -> ep.TensorType:
        if noise.shape[2:] != x.shape[1:]:
            # upscale noise
            if isinstance(noise, ep.PyTorchTensor):
                import torch

                noise = ep.astensor(
                    torch.nn.functional.upsample_bilinear(noise.raw, size=x.shape[2:])
                )
            elif isinstance(noise, ep.TensorFlowTensor):
                import tensorflow

                noise = ep.astensor(
                    tensorflow.image.resize(noise.raw, size=x.shape[1:-1])
                )
            elif isinstance(noise, ep.NumPyTensor):
                import scipy

                factors = [float(d[1]) / d[0] for d in zip(noise.shape, x.shape)]
                noise = ep.astensor(scipy.ndimage.zoom(noise.raw, factors, order=2))
            else:
                raise NotImplementedError("upsampling not yet implemented")
                # TODO: use https://github.com/google/jax/issues/862 for JAX

        return ep.clip(noise + x, -epsilon, +epsilon)

    def choice(self, a: int, size, replace: bool, p: ep.TensorType) -> ep.TensorType:
        p = p.numpy()
        x = np.random.choice(a, size, replace, p)
        return x

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: TargetedMisclassification,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        N = len(x)

        if isinstance(criterion, TargetedMisclassification):
            classes = criterion.target_classes
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            raise ValueError(
                f"expected target_classes to have shape ({N},), got {classes.shape}"
            )

        if self.reduced_dims is not None:
            if x.ndim != 4:
                raise NotImplementedError(
                    "only implemented for inputs with two spatial dimensions"
                    " (and one channel and one batch dimension)"
                )

            if self.channel_axis is None:
                channel_axis = get_channel_axis(model, x.ndim)
            else:
                channel_axis = self.channel_axis % x.ndim

            if channel_axis is None:
                raise ValueError(
                    "cannot infer the data_format from the model, please"
                    " specify channel_axis when initializing the attack"
                )

            if channel_axis == 1:
                noise_shape = (x.shape[1], *self.reduced_dims)
            elif channel_axis == 3:
                noise_shape = (*self.reduced_dims, x.shape[4])
            else:
                raise ValueError(
                    "expected 'channel_axis' to be 1 or 3, got {channel_axis}"
                )
        else:
            noise_shape = x.shape[1:]

        def is_adversarial(logits: ep.TensorType) -> ep.TensorType:
            return ep.argmax(logits, 1) == classes

        num_plateaus = ep.zeros(x, len(x))
        mutation_probability = (
            ep.ones_like(num_plateaus) * self.min_mutation_probability
        )
        mutation_range = ep.ones_like(num_plateaus) * self.min_mutation_range

        noise_pops = ep.uniform(
            x, (N, self.population, *noise_shape), -epsilon, epsilon
        )

        def calculate_fitness(logits: ep.TensorType) -> ep.TensorType:
            first = logits[range(N), classes]
            second = ep.log(ep.exp(logits).sum(1) - first)

            return first - second

        for step in range(self.steps):
            fitness, is_adv = [], []

            for i in range(self.population):
                it = self.apply_noise(x, noise_pops[:, i], epsilon)
                logits = model(it)
                print(logits.shape, classes)
                f = calculate_fitness(logits)
                a = is_adversarial(logits)
                fitness.append(f)
                is_adv.append(a)

            fitness = ep.stack(fitness)
            is_adv = ep.stack(is_adv, 1)
            elite_idxs = ep.argmax(fitness, 0)

            elite_noise = noise_pops[range(N), elite_idxs]
            is_adv = is_adv[range(N), elite_idxs]

            # early stopping
            if is_adv.all():
                return restore_type(self.apply_noise(x, elite_noise, epsilon))

            probs = ep.softmax(fitness / self.sampling_temperature, 0)
            parents_idxs = np.stack(
                [
                    self.choice(
                        self.population,
                        2 * self.population - 2,
                        replace=True,
                        p=probs[:, i],
                    )
                    for i in range(N)
                ],
                1,
            )

            mutations = [
                ep.uniform(
                    x,
                    noise_shape,
                    -mutation_range[i].item() * epsilon,
                    mutation_range[i].item() * epsilon,
                )
                for i in range(N)
            ]

            noise_pops_old = noise_pops
            noise_pops = [elite_noise]
            for i in range(0, self.population - 1):
                parents_1 = noise_pops_old[range(N), parents_idxs[2 * i]]
                parents_2 = noise_pops_old[range(N), parents_idxs[2 * i + 1]]

                # calculate crossover
                p = probs[parents_idxs[2 * i], range(N)] / (
                    probs[parents_idxs[2 * i], range(N)]
                    + probs[parents_idxs[2 * i + 1], range(N)]
                )
                p = atleast_kd(p, x.ndim)
                p = ep.tile(p, (1, *noise_shape))

                crossover_mask = ep.uniform(p, p.shape, 0, 1) < p
                children = ep.where(crossover_mask, parents_1, parents_2)

                # calculate mutation
                mutation_mask = ep.uniform(children, children.shape)
                mutation_mask = mutation_mask <= atleast_kd(
                    mutation_probability, children.ndim
                )
                children = ep.where(mutation_mask, children + mutations[i], children)

                # project back to epsilon range
                children = ep.clip(children, -epsilon, epsilon)

                noise_pops.append(children)

            noise_pops = ep.stack(noise_pops, 1)

            # TODO: increase num_plateaus if fitness does not improve
            #  for 100 consecutive steps

            # TODO: update parameters with ep.pow

            mutation_probability = ep.maximum(
                self.min_mutation_probability,
                0.5 * ep.exp(math.log(0.9) * ep.ones_like(num_plateaus) * num_plateaus),
            )
            mutation_range = ep.maximum(
                self.min_mutation_range,
                0.5 * ep.exp(math.log(0.9) * ep.ones_like(num_plateaus) * num_plateaus),
            )

        return restore_type(self.apply_noise(x, elite_noise, epsilon))
