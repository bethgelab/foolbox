import warnings
import time
import sys

from .base import Attack
from .base import generator_decorator
from ..distances import MSE, Linf
from ..criteria import Misclassification
import numpy as np
import math
from warnings import warn
import logging


class HopSkipJumpAttack(Attack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.

    Notes
    -----
    Features:
    * ability to switch between two types of distances: MSE and Linf.
    * ability to continue previous attacks by passing an instance of the
      Adversarial class
    * ability to pass an explicit starting point; especially to initialize
      a targeted attack
    * ability to pass an alternative attack used for initialization
    * ability to specify the batch size

    References
    ----------
    ..
    HopSkipJumpAttack was originally proposed by Chen, Jordan and
    Wainwright.
    It is a decision-based attack that requires access to output
    labels of a model alone.
    Paper link: https://arxiv.org/abs/1904.02144
    The implementation in Foolbox is based on Boundary Attack.

    """

    @generator_decorator
    def as_generator(
        self,
        a,
        iterations=64,
        initial_num_evals=100,
        max_num_evals=10000,
        stepsize_search="geometric_progression",
        gamma=1.0,
        starting_point=None,
        batch_size=256,
        internal_dtype=np.float64,
        log_every_n_steps=None,
        loggingLevel=logging.WARNING,
    ):
        """Applies HopSkipJumpAttack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        iterations : int
            Number of iterations to run.
        initial_num_evals: int
            Initial number of evaluations for gradient estimation.
            Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_num_evals: int
            Maximum number of evaluations for gradient estimation.
        stepsize_search: str
            How to search for stepsize; choices are 'geometric_progression',
            'grid_search'. 'geometric progression' initializes the stepsize
            by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
            until reaching the target side of the boundary. 'grid_search'
            chooses the optimal epsilon over a grid, in the scale of
            ||x_t - x||_p.
        gamma: float
            The binary search threshold theta is gamma / d^1.5 for
                   l2 attack and gamma / d^2 for linf attack.

        starting_point : `numpy.ndarray`
            Adversarial input to use as a starting point, required
            for targeted attacks.
        batch_size : int
            Batch size for model prediction.
        internal_dtype : np.float32 or np.float64
            Higher precision might be slower but is numerically more stable.
        log_every_n_steps : int
            Determines verbositity of the logging.
        loggingLevel : int
            Controls the verbosity of the logging, e.g. logging.INFO
            or logging.WARNING.

        """

        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.batch_size = batch_size
        self._starting_point = starting_point
        self.internal_dtype = internal_dtype
        self.log_every_n_steps = log_every_n_steps

        self.logger = logging.getLogger("BoundaryAttack")
        self.logger.setLevel(loggingLevel)

        # Set constraint based on the distance.
        if self._default_distance == MSE:
            self.constraint = "l2"
        elif self._default_distance == Linf:
            self.constraint = "linf"

        # Set binary search threshold.
        self.shape = a.unperturbed.shape
        self.d = np.prod(self.shape)
        if self.constraint == "l2":
            self.theta = self.gamma / (np.sqrt(self.d) * self.d)
        else:
            self.theta = self.gamma / (self.d * self.d)
        logging.info(
            "HopSkipJumpAttack optimized for {} distance".format(self.constraint)
        )

        yield from self.attack(a, iterations=iterations)

    def attack(self, a, iterations):
        """
        iterations : int
            Maximum number of iterations to run.
        """
        self.t_initial = time.time()

        # ===========================================================
        # Increase floating point precision
        # ===========================================================

        self.external_dtype = a.unperturbed.dtype

        assert self.internal_dtype in [np.float32, np.float64]
        assert self.external_dtype in [np.float32, np.float64]

        assert not (
            self.external_dtype == np.float64 and self.internal_dtype == np.float32
        )

        a.set_distance_dtype(self.internal_dtype)

        # ===========================================================
        # Construct batch decision function with binary output.
        # ===========================================================
        # decision_function = lambda x: a.forward(
        #     x.astype(self.external_dtype), strict=False)[1]
        def decision_function(x):
            outs = []
            num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[self.batch_size * j : self.batch_size * (j + 1)]
                current_batch = current_batch.astype(self.external_dtype)
                _, out = yield from a.forward(current_batch, strict=False)
                outs.append(out)
            outs = np.concatenate(outs, axis=0)
            return outs

        # ===========================================================
        # intialize time measurements
        # ===========================================================
        self.time_gradient_estimation = 0

        self.time_search = 0

        self.time_initialization = 0

        # ===========================================================
        # Initialize variables, constants, hyperparameters, etc.
        # ===========================================================

        # make sure repeated warnings are shown
        warnings.simplefilter("always", UserWarning)

        # get bounds
        bounds = a.bounds()
        self.clip_min, self.clip_max = bounds

        # ===========================================================
        # Find starting point
        # ===========================================================

        yield from self.initialize_starting_point(a)

        if a.perturbed is None:
            warnings.warn(
                "Initialization failed."
                " it might be necessary to pass an explicit starting"
                " point."
            )
            return

        self.time_initialization += time.time() - self.t_initial

        assert a.perturbed.dtype == self.external_dtype
        # get original and starting point in the right format
        original = a.unperturbed.astype(self.internal_dtype)
        perturbed = a.perturbed.astype(self.internal_dtype)

        # ===========================================================
        # Iteratively refine adversarial
        # ===========================================================
        t0 = time.time()

        # Project the initialization to the boundary.
        perturbed, dist_post_update = yield from self.binary_search_batch(
            original, np.expand_dims(perturbed, 0), decision_function
        )

        dist = self.compute_distance(perturbed, original)

        distance = a.distance.value
        self.time_search += time.time() - t0

        # log starting point
        self.log_step(0, distance)

        for step in range(1, iterations + 1):

            t0 = time.time()

            # ===========================================================
            # Gradient direction estimation.
            # ===========================================================
            # Choose delta.
            delta = self.select_delta(dist_post_update, step)

            # Choose number of evaluations.
            num_evals = int(
                min([self.initial_num_evals * np.sqrt(step), self.max_num_evals])
            )

            # approximate gradient.
            gradf = yield from self.approximate_gradient(
                decision_function, perturbed, num_evals, delta
            )

            if self.constraint == "linf":
                update = np.sign(gradf)
            else:
                update = gradf
            t1 = time.time()
            self.time_gradient_estimation += t1 - t0

            # ===========================================================
            # Update, and binary search back to the boundary.
            # ===========================================================
            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilon = yield from self.geometric_progression_for_stepsize(
                    perturbed, update, dist, decision_function, step
                )

                # Update the sample.
                perturbed = np.clip(
                    perturbed + epsilon * update, self.clip_min, self.clip_max
                )

                # Binary search to return to the boundary.
                perturbed, dist_post_update = yield from self.binary_search_batch(
                    original, perturbed[None], decision_function
                )

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = np.clip(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = yield from decision_function(perturbeds)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum
                    # distance after binary search.
                    perturbed, dist_post_update = yield from self.binary_search_batch(
                        original, perturbeds[idx_perturbed], decision_function
                    )
            t2 = time.time()

            self.time_search += t2 - t1

            # compute new distance.
            dist = self.compute_distance(perturbed, original)

            # ===========================================================
            # Log the step
            # ===========================================================
            # Using foolbox definition of distance for logging.
            if self.constraint == "l2":
                distance = dist ** 2 / self.d / (self.clip_max - self.clip_min) ** 2
            elif self.constraint == "linf":
                distance = dist / (self.clip_max - self.clip_min)
            message = " (took {:.5f} seconds)".format(t2 - t0)
            self.log_step(step, distance, message)
            sys.stdout.flush()

        # ===========================================================
        # Log overall runtime
        # ===========================================================

        self.log_time()

    # ===============================================================
    #
    # Other methods
    #
    # ===============================================================

    def initialize_starting_point(self, a):
        starting_point = self._starting_point

        if a.perturbed is not None:
            print(
                "Attack is applied to a previously found adversarial."
                " Continuing search for better adversarials."
            )
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    "Ignoring starting_point parameter because the attack"
                    " is applied to a previously found adversarial."
                )
            return

        if starting_point is not None:
            yield from a.forward_one(starting_point)
            assert (
                a.perturbed is not None
            ), "Invalid starting point provided. Please provide a starting point that is adversarial."
            return

        """
        Apply BlendedUniformNoiseAttack if without
        initialization.
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0

        while True:
            random_noise = np.random.uniform(
                self.clip_min, self.clip_max, size=self.shape
            )
            _, success = yield from a.forward_one(
                random_noise.astype(self.external_dtype)
            )
            num_evals += 1
            if success:
                break
            if num_evals > 1e4:
                return

        # Binary search to minimize l2 distance to the original input.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * a.unperturbed + mid * random_noise
            _, success = yield from a.forward_one(blended.astype(self.external_dtype))
            if success:
                high = mid
            else:
                low = mid

    def compute_distance(self, x1, x2):
        if self.constraint == "l2":
            return np.linalg.norm(x1 - x2)
        elif self.constraint == "linf":
            return np.max(abs(x1 - x2))

    def project(self, unperturbed, perturbed_inputs, alphas):
        """ Projection onto given l2 / linf balls in a batch. """
        alphas_shape = [len(alphas)] + [1] * len(self.shape)
        alphas = alphas.reshape(alphas_shape)
        if self.constraint == "l2":
            projected = (1 - alphas) * unperturbed + alphas * perturbed_inputs
        elif self.constraint == "linf":
            projected = np.clip(
                perturbed_inputs, unperturbed - alphas, unperturbed + alphas
            )
        return projected

    def binary_search_batch(self, unperturbed, perturbed_inputs, decision_function):
        """ Binary search to approach the boundary. """

        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = np.array(
            [
                self.compute_distance(unperturbed, perturbed_x)
                for perturbed_x in perturbed_inputs
            ]
        )

        # Choose upper thresholds in binary searchs based on constraint.
        if self.constraint == "linf":
            highs = dists_post_update
            # Stopping criteria.
            thresholds = dists_post_update * self.theta
        else:
            highs = np.ones(len(perturbed_inputs))
            thresholds = self.theta

        lows = np.zeros(len(perturbed_inputs))

        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            decisions = yield from decision_function(mid_inputs)
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = np.array(
            [self.compute_distance(unperturbed, out) for out in out_inputs]
        )
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        return out, dist

    def select_delta(self, dist_post_update, current_iteration):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.constraint == "l2":
                delta = np.sqrt(self.d) * self.theta * dist_post_update
            elif self.constraint == "linf":
                delta = self.d * self.theta * dist_post_update

        return delta

    def approximate_gradient(self, decision_function, sample, num_evals, delta):
        """ Gradient direction estimation """
        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)
        if self.constraint == "l2":
            rv = np.random.randn(*noise_shape)
        elif self.constraint == "linf":
            rv = np.random.uniform(low=-1, high=1, size=noise_shape)

        axis = tuple(range(1, 1 + len(self.shape)))
        rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
        perturbed = sample + delta * rv
        perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = yield from decision_function(perturbed)
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.astype(self.internal_dtype).reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        vals = fval if abs(np.mean(fval)) == 1.0 else fval - np.mean(fval)
        gradf = np.mean(vals * rv, axis=0)

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

    def geometric_progression_for_stepsize(
        self, x, update, dist, decision_function, current_iteration
    ):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = np.clip(x + epsilon * update, self.clip_min, self.clip_max)
            success = (yield from decision_function(updated[None]))[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover

        return epsilon

    def log_step(self, step, distance, message="", always=False):
        if self.log_every_n_steps is None or self.log_every_n_steps == np.inf:
            return
        if not always and step % self.log_every_n_steps != 0:
            return
        logging.info("Step {}: {:.5e} {}".format(step, distance, message))

    def log_time(self):
        t_total = time.time() - self.t_initial
        rel_initialization = self.time_initialization / t_total
        rel_gradient_estimation = self.time_gradient_estimation / t_total
        rel_search = self.time_search / t_total

        self.printv("Time since beginning: {:.5f}".format(t_total))
        self.printv(
            "   {:2.1f}% for initialization ({:.5f})".format(
                rel_initialization * 100, self.time_initialization
            )
        )
        self.printv(
            "   {:2.1f}% for gradient estimation ({:.5f})".format(
                rel_gradient_estimation * 100, self.time_gradient_estimation
            )
        )
        self.printv(
            "   {:2.1f}% for search ({:.5f})".format(rel_search * 100, self.time_search)
        )

    def printv(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)


def BoundaryAttackPlusPlus(
    model=None, criterion=Misclassification(), distance=MSE, threshold=None
):
    warn("BoundaryAttackPlusPlus is deprecated; use HopSkipJumpAttack.")
    return HopSkipJumpAttack(model, criterion, distance, threshold)
