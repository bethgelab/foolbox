import numpy as np
import eagerpy as ep
from collections.abc import Callable
import logging

from ..utils import flatten
from ..utils import atleast_kd
from . import LinearSearchBlendedUniformNoiseAttack
from ..tensorboard import TensorBoard


def misclassification(
    inputs: ep.Tensor, labels: ep.Tensor, perturbed: ep.Tensor, logits: ep.Tensor
) -> ep.Tensor:
    classes = logits.argmax(axis=-1)
    return classes != labels


class TargetedMisclassification:
    def __init__(self, target_classes: ep.Tensor):
        self.target_classes = target_classes

    def __call__(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor,
        perturbed: ep.Tensor,
        logits: ep.Tensor,
    ) -> ep.Tensor:
        classes = logits.argmax(axis=-1)
        return classes == self.target_classes


class BoundaryAttack:
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        inputs,
        labels,
        *,
        starting_points=None,
        init_attack=None,
        criterion: Callable = misclassification,
        steps=25000,
        spherical_step=1e-2,
        source_step=1e-2,
        source_step_convergance=1e-7,
        step_adaptation=1.5,
        tensorboard=False,
        update_stats_every_k=10,
    ):
        """Boundary Attack

        Differences to the original reference implementation:
        * We do not perform internal operations with float64
        * The samples within a batch can currently influence each other a bit
        * We don't perform the additional convergence confirmation
        * The success rate tracking changed a bit
        * Some other changes due to batching and merged loops

        Parameters
        ----------
        criterion : Callable
            A callable that returns true if the given logits of perturbed
            inputs should be considered adversarial w.r.t. to the given labels
            and unperturbed inputs.
        tensorboard : str
            The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        """
        tb = TensorBoard(logdir=tensorboard)

        originals = ep.astensor(inputs)
        labels = ep.astensor(labels)

        def is_adversarial(p: ep.Tensor) -> ep.Tensor:
            """For each input in x, returns true if it is an adversarial for
            the given model and criterion"""
            logits = ep.astensor(self.model.forward(p.tensor))
            return criterion(originals, labels, p, logits)

        if starting_points is None:
            if init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack.__name__} for initialization."
                )
            starting_points = init_attack(self.model)(inputs, labels)

        best_advs = ep.astensor(starting_points)
        assert is_adversarial(best_advs).all()

        N = len(originals)
        ndim = originals.ndim
        spherical_steps = ep.ones(originals, N) * spherical_step
        source_steps = ep.ones(originals, N) * source_step

        tb.scalar("batchsize", N, 0)

        # create two queues for each sample to track success rates
        # (used to update the hyper parameters)
        stats_spherical_adversarial = ArrayQueue(maxlen=100, N=N)
        stats_step_adversarial = ArrayQueue(maxlen=30, N=N)

        bounds = self.model.bounds()

        for step in range(1, steps + 1):
            converged = source_steps < source_step_convergance
            if converged.all():
                break
            converged = atleast_kd(converged, ndim)

            # TODO: performance: ignore those that have converged
            # (we could select the non-converged ones, but we currently
            # cannot easily invert this in the end using EagerPy)

            unnormalized_source_directions = originals - best_advs
            source_norms = l2norms(unnormalized_source_directions)
            source_directions = unnormalized_source_directions / atleast_kd(
                source_norms, ndim
            )

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % update_stats_every_k == 0

            candidates, spherical_candidates = draw_proposals(
                bounds,
                originals,
                best_advs,
                unnormalized_source_directions,
                source_directions,
                source_norms,
                spherical_steps,
                source_steps,
            )
            candidates.dtype == originals.dtype
            spherical_candidates.dtype == spherical_candidates.dtype

            is_adv = is_adversarial(candidates)

            if check_spherical_and_update_stats:
                spherical_is_adv = is_adversarial(spherical_candidates)
                stats_spherical_adversarial.append(spherical_is_adv)
                # TODO: algorithm: the original implementation ignores those samples
                # for which spherical is not adversarial and continues with the
                # next iteration -> we estimate different probabilities (conditional vs. unconditional)
                # TODO: thoughts: should we always track this because we compute it anyway
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None

            # in theory, we are closer per construction
            # but limited numerical precision might break this
            distances = l2norms(originals - candidates)
            closer = distances < source_norms
            is_best_adv = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)

            cond = converged.logical_not().logical_and(is_best_adv)
            best_advs = ep.where(cond, candidates, best_advs)

            tb.probability("converged", converged, step)
            tb.scalar("updated_stats", check_spherical_and_update_stats, step)
            tb.histogram("norms", source_norms, step)
            tb.probability("is_adv", is_adv, step)
            if spherical_is_adv is not None:
                tb.probability("spherical_is_adv", spherical_is_adv, step)
            tb.histogram("candidates/distances", distances, step)
            tb.probability("candidates/closer", closer, step)
            tb.probability("candidates/is_best_adv", is_best_adv, step)
            tb.probability("new_best_adv_including_converged", is_best_adv, step)
            tb.probability("new_best_adv", cond, step)

            if check_spherical_and_update_stats:
                full = stats_spherical_adversarial.isfull()
                tb.probability("spherical_stats/full", full, step)
                if full.any():
                    probs = stats_spherical_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.5, full)
                    spherical_steps = ep.where(
                        cond1, spherical_steps * step_adaptation, spherical_steps
                    )
                    source_steps = ep.where(
                        cond1, source_steps * step_adaptation, source_steps
                    )
                    cond2 = ep.logical_and(probs < 0.2, full)
                    spherical_steps = ep.where(
                        cond2, spherical_steps / step_adaptation, spherical_steps
                    )
                    source_steps = ep.where(
                        cond2, source_steps / step_adaptation, source_steps
                    )
                    stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
                    tb.conditional_mean(
                        "spherical_stats/isfull/success_rate/mean", probs, full, step,
                    )
                    tb.probability_ratio(
                        "spherical_stats/isfull/too_linear", cond1, full, step,
                    )
                    tb.probability_ratio(
                        "spherical_stats/isfull/too_nonlinear", cond2, full, step,
                    )

                full = stats_step_adversarial.isfull()
                tb.probability("step_stats/full", full, step)
                if full.any():
                    probs = stats_step_adversarial.mean()
                    # TODO: algorithm: changed the two values because we are currently tracking p(source_step_sucess)
                    # instead of p(source_step_success | spherical_step_sucess) that was tracked before
                    cond1 = ep.logical_and(probs > 0.25, full)
                    source_steps = ep.where(
                        cond1, source_steps * step_adaptation, source_steps
                    )
                    cond2 = ep.logical_and(probs < 0.1, full)
                    source_steps = ep.where(
                        cond2, source_steps / step_adaptation, source_steps
                    )
                    stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
                    tb.conditional_mean(
                        "step_stats/isfull/success_rate/mean", probs, full, step,
                    )
                    tb.probability_ratio(
                        "step_stats/isfull/success_rate_too_high", cond1, full, step,
                    )
                    tb.probability_ratio(
                        "step_stats/isfull/success_rate_too_low", cond2, full, step,
                    )

            tb.histogram("spherical_step", spherical_steps, step)
            tb.histogram("source_step", source_steps, step)
        tb.close()
        return best_advs.tensor


class ArrayQueue:
    def __init__(self, maxlen, N):
        # we use NaN as an indicator for missing data
        self.data = np.full((maxlen, N), np.nan)
        self.next = 0
        # used to infer the correct framework because this class uses NumPy
        self.tensor = None

    @property
    def maxlen(self):
        return self.data.shape[0]

    @property
    def N(self):
        return self.data.shape[1]

    def append(self, x: ep.Tensor):
        if self.tensor is None:
            self.tensor = x
        x = x.numpy()
        assert x.shape == (self.N,)
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims: ep.Tensor):
        if self.tensor is None:
            self.tensor = dims
        dims = dims.numpy()
        assert dims.shape == (self.N,)
        assert dims.dtype == np.bool
        self.data[:, dims] = np.nan

    def mean(self):
        result = np.nanmean(self.data, axis=0)
        return ep.from_numpy(self.tensor, result)

    def isfull(self):
        result = ~np.isnan(self.data).any(axis=0)
        return ep.from_numpy(self.tensor, result)


def l2norms(x):
    return flatten(x).square().sum(axis=-1).sqrt()


def draw_proposals(
    bounds,
    originals: ep.Tensor,
    perturbed: ep.Tensor,
    unnormalized_source_directions: ep.Tensor,
    source_directions: ep.Tensor,
    source_norms: ep.Tensor,
    spherical_steps: ep.Tensor,
    source_steps: ep.Tensor,
):
    # remember the actual shape
    shape = originals.shape
    assert perturbed.shape == shape
    assert unnormalized_source_directions.shape == shape
    assert source_directions.shape == shape

    # flatten everything to (batch, size)
    originals = flatten(originals)
    perturbed = flatten(perturbed)
    unnormalized_source_directions = flatten(unnormalized_source_directions)
    source_directions = flatten(source_directions)
    N, D = originals.shape

    assert source_norms.shape == (N,)
    assert spherical_steps.shape == (N,)
    assert source_steps.shape == (N,)

    # draw from an iid Gaussian (we can share this across the whole batch)
    eta = ep.normal(perturbed, (D, 1))

    # make orthogonal (source_directions are normalized)
    eta = eta.T - ep.matmul(source_directions, eta) * source_directions
    assert eta.shape == (N, D)

    # rescale
    norms = l2norms(eta)
    assert norms.shape == (N,)
    eta = eta * atleast_kd(spherical_steps * source_norms / norms, eta.ndim)

    # project on the sphere using Pythagoras
    distances = atleast_kd((spherical_steps.square() + 1).sqrt(), eta.ndim)
    directions = eta - unnormalized_source_directions
    spherical_candidates = originals + directions / distances

    # clip
    min_, max_ = bounds
    spherical_candidates = spherical_candidates.clip(min_, max_)

    # step towards the original inputs
    new_source_directions = originals - spherical_candidates
    assert new_source_directions.ndim == 2
    new_source_directions_norms = l2norms(new_source_directions)

    # length if spherical_candidates would be exactly on the sphere
    lengths = source_steps * source_norms

    # length including correction for numerical deviation from sphere
    lengths = lengths + new_source_directions_norms - source_norms

    # make sure the step size is positive
    lengths = ep.maximum(lengths, 0)

    # normalize the length
    lengths = lengths / new_source_directions_norms
    lengths = atleast_kd(lengths, new_source_directions.ndim)

    candidates = spherical_candidates + lengths * new_source_directions

    # clip
    candidates = candidates.clip(min_, max_)

    # restore shape
    candidates = candidates.reshape(shape)
    spherical_candidates = spherical_candidates.reshape(shape)
    return candidates, spherical_candidates
