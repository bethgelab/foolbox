from typing import Union, Optional, Any, List, Tuple
import numpy as np
import eagerpy as ep

from ..devutils import atleast_kd

from ..models import Model

from ..criteria import Criterion, TargetedMisclassification

from ..distances import l0

from .base import MinimizationAttack
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import get_channel_axis
from .base import raise_if_kwargs


class LocalSearchAttack(MinimizationAttack):
    """A black-box attack based on the idea of greedy local search. [#Naro]

    Args:
        steps : An upper bound on the number of iterations.
        r : Perturbation parameter that controls the cyclic perturbation;
            must be in [0, 2].
        p : Perturbation parameter that controls the pixel sensitivity
            estimation.
        d : The half side length of the neighborhood square.
        t : The number of pixels perturbed at each round.
        channel_axis : Index of the channel axis in the input data.


    References:
        .. [#Naro] Nina Narodytska, Shiva Prasad Kasiviswanathan, "Simple
           Black-Box Adversarial Perturbations for Deep Networks",
           https://arxiv.org/abs/1612.06299
    """

    def __init__(
        self,
        *,
        steps: int = 150,
        r: float = 1.5,
        p: float = 10.0,
        d: int = 5,
        t: int = 5,
        max_initial_pixels: int = 150,
        channel_axis: Optional[int] = None,
    ):
        self.steps = steps
        self.r = r
        self.p = p
        self.d = d
        self.t = t
        self.max_initial_pixels = max_initial_pixels
        self.channel_axis = channel_axis

    distance = l0

    def run(  # noqa: C901
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if isinstance(criterion, TargetedMisclassification):
            classes = criterion.target_classes
        else:
            raise ValueError("unsupported criterion")

        if x.ndim != 4:
            raise NotImplementedError(
                "only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)"
            )

        if self.channel_axis is None:
            channel_axis = get_channel_axis(model, x.ndim)
        else:
            channel_axis = self.channel_axis % x.ndim

        if channel_axis is None:
            raise ValueError(
                "cannot infer the data_format from the model, please specify"
                " channel_axis when initializing the attack"
            )

        if channel_axis == 1:
            h, w = x.shape[2:4]
        elif channel_axis == 3:
            h, w = x.shape[1:3]
        else:
            raise ValueError("expected 'channel_axis' to be 1 or 3, got {channel_axis}")

        min_, max_ = model.bounds
        N = len(x)

        def normalize(im: ep.TensorType) -> Tuple[ep.TensorType, float, float]:
            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2

            return im, LB, UB

        def unnormalize(im: ep.TensorType) -> ep.TensorType:
            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2

            return im

        x, LB, UB = normalize(x)

        def pert(  # type: ignore
            x: ep.TensorType,
            p: float,
            px_x: "np.ndarray",
            px_y: "np.ndarray",
            channel_axis: int,
        ) -> ep.TensorType:
            location_lst = [ep.arange(x, len(x)), px_x, px_y]
            location_lst.insert(channel_axis, slice(None))
            location = tuple(location_lst)

            delta_values = p * ep.sign(x[location])

            delta = ep.zeros_like(x)
            location = ep.index[location]
            delta = ep.index_update(delta, location, delta_values)

            return delta + x

        def random_locations() -> List["np.ndarray"]:  # type: ignore
            n = int(0.1 * h * w)
            n = min(n, self.max_initial_pixels)
            locations = np.array([np.random.permutation(h * w)[:n] for _ in range(N)])
            p_x = locations % w
            p_y = locations % h
            pxpy = np.stack((p_x, p_y), -1)

            return list(pxpy)

        def cyclic(r: float, x_xy: ep.TensorType) -> ep.TensorType:
            result = r * x_xy

            result = ep.where(result < LB, result + (UB - LB), result)
            result = ep.where(result > UB, result - (UB - LB), result)

            return result

        def batch_cyclic(  # type: ignore
            x: ep.TensorType, px_py_star: List["np.ndarray"], channel_axis: int
        ) -> ep.TensorType:
            for n in range(N):
                # Generation of new perturbed input x
                for px_x, px_y in px_py_star[n]:
                    # TODO: replace this loop with a call for all channels
                    #  when eagerpy's index_update for tensorflow supports slices
                    for b in range(x.shape[channel_axis]):
                        location_lst = [n, px_x, px_y]
                        location_lst.insert(channel_axis, b)
                        location = tuple(location_lst)
                        x = ep.index_update(x, location, cyclic(self.r, x[location]))

            return x

        px_py = random_locations()

        x_adv = x

        for step in range(self.steps):
            # chose a random subset of the px_py indices
            for i in range(N):
                px_py[i] = px_py[i][
                    np.random.permutation(len(px_py[i]))[: self.max_initial_pixels]
                ]

            # calculate new px_py_star
            max_length_px_py = max([len(it) for it in px_py])
            scores_list: List[List[float]] = [[] for _ in range(N)]
            for length in range(max_length_px_py):
                mask = [len(it) >= length for it in px_py]

                x_masked = x[mask]
                classes_masked = classes[mask]
                px_py_masked = np.array(px_py)[mask]

                px_x_masked = [it[length, 0] for it in px_py_masked]
                px_y_masked = [it[length, 1] for it in px_py_masked]

                perturbed_x_masked = pert(
                    x_masked, self.p, px_x_masked, px_y_masked, channel_axis
                )

                logits_masked = model(unnormalize(perturbed_x_masked))
                score_masked = ep.softmax(logits_masked, -1)[
                    range(N), classes_masked
                ].numpy()

                # apply something like 'indexing with index lists' for lists
                j = 0
                for i in range(N):
                    if mask[i]:
                        scores_list[i] = np.append(scores_list[i], score_masked[j])
                        j += 1

            # for every sample in the batch, get the t best scoring items
            indices_list = [np.argsort(lst)[-self.t :] for lst in scores_list]
            px_py_star = [lst[idxs] for lst, idxs in zip(px_py, indices_list)]

            # now calculate new updated I by perturbing it at the locations
            # px_py_star
            x = batch_cyclic(x, px_py_star, channel_axis)

            # Check whether the perturbed input x is an adversarial input
            x_unnorm = unnormalize(x)
            is_adv = is_adversarial(x_unnorm)
            x_adv = ep.where(atleast_kd(is_adv, x.ndim), x_unnorm, x_adv)

            if is_adv.any():
                break

            # Update a neighborhood of pixel locations for the next round
            for n in range(N):
                px_py_candidates = [
                    (x, y)
                    for _a, _b in px_py[n]
                    for x in range(_a - self.d, _a + self.d + 1)
                    for y in range(_b - self.d, _b + self.d + 1)
                ]

                px_py_candidates = [
                    (x, y) for x, y in px_py_candidates if 0 <= x < w and 0 <= y < h
                ]
                px_py_candidates = list(set(px_py_candidates))
                px_py[n] = np.array(px_py_candidates)

        return restore_type(x_adv)
