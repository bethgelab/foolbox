from typing import Optional, Any, Tuple, Union, List
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


def rescale_jax(img: "np.ndarray", resize_rates: Tuple[float, float]) -> "np.ndarray":
    # img must be in channel_last format

    # modified according to https://github.com/google/jax/issues/862
    import jax.numpy as np

    def interpolate_bilinear(
        im: "np.ndarray", rows: "np.ndarray", cols: "np.ndarray"
    ) -> "np.ndarray":
        # based on http://stackoverflow.com/a/12729229
        col_lo = np.floor(cols).astype(int)
        col_hi = col_lo + 1
        row_lo = np.floor(rows).astype(int)
        row_hi = row_lo + 1

        def cclip(cols: "np.ndarray") -> "np.ndarray":
            return np.clip(cols, 0, ncols - 1)

        def rclip(rows: "np.ndarray") -> "np.ndarray":
            return np.clip(rows, 0, nrows - 1)

        nrows, ncols = im.shape[-3:-1]

        Ia = im[..., rclip(row_lo), cclip(col_lo), :]
        Ib = im[..., rclip(row_hi), cclip(col_lo), :]
        Ic = im[..., rclip(row_lo), cclip(col_hi), :]
        Id = im[..., rclip(row_hi), cclip(col_hi), :]

        wa = np.expand_dims((col_hi - cols) * (row_hi - rows), -1)
        wb = np.expand_dims((col_hi - cols) * (rows - row_lo), -1)
        wc = np.expand_dims((cols - col_lo) * (row_hi - rows), -1)
        wd = np.expand_dims((cols - col_lo) * (rows - row_lo), -1)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    nrows, ncols = img.shape[-3:-1]
    deltas = (0.5 / resize_rates[0], 0.5 / resize_rates[1])

    rows = np.linspace(deltas[0], nrows - deltas[0], np.int32(resize_rates[0] * nrows))
    cols = np.linspace(deltas[1], ncols - deltas[1], np.int32(resize_rates[1] * ncols))
    rows_grid, cols_grid = np.meshgrid(rows, cols, indexing="ij")

    img_resize_vec = interpolate_bilinear(img, rows_grid.flatten(), cols_grid.flatten())
    img_resize = img_resize_vec.reshape(
        img.shape[:-3] + (len(rows), len(cols)) + img.shape[-1:]
    )

    return img_resize


def rescale_numpy(x: "np.ndarray", target_shape: List[int]) -> "np.ndarray":
    import scipy

    factors = [float(d[1]) / d[0] for d in zip(x.shape, target_shape)]
    return scipy.ndimage.zoom(x, factors, order=2)


def rescale_tensorflow(x: "np.ndarray", target_shape: List[int]) -> "np.ndarray":
    import tensorflow

    return tensorflow.image.resize(x, size=target_shape[1:-1])


def rescale_pytorch(x: Any, target_shape: List[int]) -> Any:
    import torch

    return torch.nn.functional.interpolate(
        x, size=target_shape[2:], mode="bilinear", align_corners=True
    )


def swap_axes(x: ep.TensorType, dim0: int, dim1: int):
    assert dim0 < x.ndim
    assert dim1 < x.ndim

    axes = list(range(x.ndim))
    axes[dim0] = dim1
    axes[dim1] = dim0

    return ep.transpose(x, tuple(axes))


def rescale_images(
    x: ep.TensorType, target_shape: Union[Tuple[int, ...], List[int]], channel_axis: int
) -> ep.TensorType:
    target_shape = list(target_shape)

    if isinstance(x, ep.PyTorchTensor):
        if channel_axis != 1:
            x = swap_axes(x, channel_axis, 1)

        x = ep.astensor(rescale_pytorch(x.raw, target_shape))

        if channel_axis != 1:
            x = swap_axes(x, channel_axis, 1)

    elif isinstance(x, ep.TensorFlowTensor):
        if channel_axis != x.ndim:
            x = swap_axes(x, channel_axis, x.ndim - 1)

        x = ep.astensor(rescale_tensorflow(x.raw, target_shape))

        if channel_axis != x.ndim:
            x = swap_axes(x, channel_axis, x.ndim - 1)

    elif isinstance(x, ep.NumPyTensor):
        x = ep.astensor(rescale_numpy(x.raw, target_shape))

    elif isinstance(x, ep.JAXTensor):
        if channel_axis != -1:
            x = swap_axes(x, channel_axis, -1)
            target_shape[channel_axis], target_shape[-1] = (
                target_shape[-1],
                target_shape[channel_axis],
            )

        factors = (target_shape[1] / x.shape[1], target_shape[2] / x.shape[2])

        x = ep.astensor(rescale_jax(x.raw, factors))
        if channel_axis != -1:
            x = swap_axes(x, channel_axis, -1)

    return x


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
        self, x: ep.TensorType, noise: ep.TensorType, epsilon: float, channel_axis: int
    ) -> ep.TensorType:
        if noise.shape != x.shape:
            # upscale noise

            noise = rescale_images(noise, x.shape, channel_axis)

        return ep.clip(noise + x, -epsilon, +epsilon)

    def choice(
        self, a: int, size: Union[int, ep.TensorType], replace: bool, p: ep.TensorType
    ) -> Any:
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

        noise_shape: Union[Tuple[int, int, int, int], Tuple[int, ...]]
        channel_axis: int
        if self.reduced_dims is not None:
            if x.ndim != 4:
                raise NotImplementedError(
                    "only implemented for inputs with two spatial dimensions"
                    " (and one channel and one batch dimension)"
                )

            if self.channel_axis is None:
                maybe_axis = get_channel_axis(model, x.ndim)
                if maybe_axis is None:
                    raise ValueError(
                        "cannot infer the data_format from the model, please"
                        " specify channel_axis when initializing the attack"
                    )
                else:
                    channel_axis = maybe_axis
            else:
                channel_axis = self.channel_axis % x.ndim

            if channel_axis == 1:
                noise_shape = (x.shape[1], *self.reduced_dims)
            elif channel_axis == 3:
                noise_shape = (*self.reduced_dims, x.shape[3])
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
            fitness_l, is_adv_l = [], []

            for i in range(self.population):
                it = self.apply_noise(x, noise_pops[:, i], epsilon, channel_axis)
                logits = model(it)
                f = calculate_fitness(logits)
                a = is_adversarial(logits)
                fitness_l.append(f)
                is_adv_l.append(a)

            fitness = ep.stack(fitness_l)
            is_adv = ep.stack(is_adv_l, 1)
            elite_idxs = ep.argmax(fitness, 0)

            elite_noise = noise_pops[range(N), elite_idxs]
            is_adv = is_adv[range(N), elite_idxs]

            # early stopping
            if is_adv.all():
                return restore_type(
                    self.apply_noise(x, elite_noise, epsilon, channel_axis)
                )

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

            new_noise_pops = [elite_noise]
            for i in range(0, self.population - 1):
                parents_1 = noise_pops[range(N), parents_idxs[2 * i]]
                parents_2 = noise_pops[range(N), parents_idxs[2 * i + 1]]

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

                new_noise_pops.append(children)

            noise_pops = ep.stack(new_noise_pops, 1)

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

        return restore_type(self.apply_noise(x, elite_noise, epsilon, channel_axis))
