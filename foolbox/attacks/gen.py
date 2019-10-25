from __future__ import division
import numpy as np
import logging
from scipy.ndimage import zoom

from .base import Attack
from .base import generator_decorator
from ..utils import softmax


class GenAttack(Attack):
    """The GenAttack introduced in [1]_.

    This attack is performs a genetic search in order to find an adversarial
    perturbation in a black-box scenario in as few queries as possible.

    References
    ----------
    .. [1] Moustafa Alzantot, Yash Sharma, Supriyo Chakraborty, Huan Zhang,
           Cho-Jui Hsieh, Mani Srivastava,
           "GenAttack: Practical Black-box Attacks with Gradient-Free
           Optimization",
            https://arxiv.org/abs/1607.02533
    """

    @generator_decorator
    def as_generator(
        self,
        a,
        generations=10,
        alpha=1.0,
        p=5e-2,
        N=10,
        tau=0.1,
        search_shape=None,
        epsilon=0.3,
        binary_search=20,
    ):
        """A black-box attack based on genetic algorithms.
        Can either try to find an adversarial perturbation for a fixed epsilon
        distance or perform a binary search over epsilon values in order to find
        a minimal perturbation.
        Parameters
        ----------
        inputs : `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model.
        labels : `numpy.ndarray`
            Class labels of the inputs as a vector of integers in [0, number of classes).
        unpack : bool
            If true, returns the adversarial inputs as an array, otherwise returns Adversarial objects.
        generations : int
            Number of generations, i.e. iterations, in the genetic algorithm.
        alpha : float
            Mutation-range.
        p : float
            Mutation probability.
        N : int
            Population size of the genetic algorithm.
        tau: float
            Temperature for the softmax sampling used to determine the parents
            of the new crossover.
        search_shape : tuple (default: None)
            Set this to a smaller image shape than the true shape to search in
            a smaller input space. The input will be scaled using a linear
            interpolation to match the required input shape of the model.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and using their
            values to start the search. If False, hyperparameters are not
            optimized. Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        """

        assert a.target_class is not None, "GenAttack is a targeted attack."

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            yield from self._run_binary_search(
                a, epsilon, k, generations, alpha, p, N, tau, search_shape
            )
            return
        else:
            yield from self._run_one(
                a, generations, alpha, p, N, tau, search_shape, epsilon
            )
            return

    def _run_one(self, a, generations, alpha, rho, N, tau, search_shape, epsilon):

        min_, max_ = a.bounds()

        x = a.unperturbed

        search_shape = x.shape if search_shape is None else search_shape

        assert len(search_shape) == len(x.shape), (
            "search_shape must have the same rank as the original " "image's shape"
        )

        def get_perturbed(population_noises):
            if population_noises[0].shape != x.shape:
                factors = [float(d[1]) / d[0] for d in zip(search_shape, x.shape)]
                population_noises = zoom(population_noises, zoom=(1, *factors), order=2)

            # project into epsilon ball and valid bounds
            return np.clip(
                np.clip(population_noises, -epsilon, epsilon) + x, min_, max_
            )

        population = np.random.uniform(-epsilon, +epsilon, (N, *search_shape)).astype(
            x.dtype
        )

        for g in range(generations):
            x_perturbed = get_perturbed(population)

            probs, is_adversarial = [], []
            # TODO: Replace this with a single call to a.forward(...) once this
            #   is implemented
            for it in x_perturbed:
                l, i = yield from a.forward_one(it)
                probs.append(softmax(l))
                is_adversarial.append(i)
            probs = np.array(probs)
            masked_probs = probs.copy()
            masked_probs[:, a.target_class] = 0

            fitnesses = np.log(probs[:, a.target_class] + 1e-30) - np.log(
                np.sum(masked_probs, 1) + 1e-30
            )

            # find elite member
            elite_idx = np.argmax(fitnesses)

            # TODO: Does this make sense in our framework? We can just ignore
            #  this and use the minimal distortion tracked by the a

            # elite member already is adversarial example
            if is_adversarial[elite_idx]:
                yield from a.forward_one(x_perturbed[elite_idx])
                return True

            next_population = population.copy()

            mutation_probabilities = softmax(fitnesses / tau)

            # determine crossover between two parents
            parents_idx = np.random.choice(
                N, 2 * N - 2, replace=True, p=mutation_probabilities
            ).reshape(2, -1)
            p = fitnesses[parents_idx[0]] / (
                fitnesses[parents_idx[0]] + fitnesses[parents_idx[1]]
            )
            p = p.reshape(-1, *([1] * (len(population.shape) - 1)))
            crossover = (
                p * population[parents_idx[0]] + (1 - p) * population[parents_idx[1]]
            )

            # determine new mutation in this generation
            b = (np.random.uniform(0, 1, (N - 1, 1, 1, 1)) < rho).astype(np.float32)
            mutation = b * np.random.uniform(
                -alpha * epsilon, +alpha * epsilon, (N - 1, *search_shape)
            )

            next_population[1:] = crossover + mutation

            population = next_population

        return False

    def _run_binary_search(
        self, a, epsilon, k, generations, alpha, p, N, tau, search_shape
    ):
        def try_epsilon(epsilon):
            success = yield from self._run_one(
                a, generations, alpha, p, N, tau, search_shape, epsilon
            )
            return success

        for i in range(k):
            success = yield from try_epsilon(epsilon)
            if success:
                logging.info("successful for eps = {}".format(epsilon))
                break
            logging.info("not successful for eps = {}".format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning("exponential search failed")
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            success = yield from try_epsilon(epsilon)
            if success:
                good = epsilon
                logging.info("successful for eps = {}".format(epsilon))
            else:
                bad = epsilon
                logging.info("not successful for eps = {}".format(epsilon))
