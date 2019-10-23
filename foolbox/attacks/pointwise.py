import warnings
import logging

from .base import Attack
from .base import generator_decorator
from .saltandpepper import SaltAndPepperNoiseAttack
from .. import rng


class PointwiseAttack(Attack):
    """Starts with an adversarial and performs a binary search between
    the adversarial and the original for each dimension of the input
    individually.

    References
    ----------
    .. [1] L. Schott, J. Rauber, M. Bethge, W. Brendel: "Towards the first
           adversarially robust neural network model on MNIST", ICLR (2019)
           https://arxiv.org/abs/1805.09190

    """

    @generator_decorator
    def as_generator(self, a, starting_point=None, initialization_attack=None):

        """Starts with an adversarial and performs a binary search between
        the adversarial and the original for each dimension of the input
        individually.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        starting_point : `numpy.ndarray`
            Adversarial input to use as a starting point, in particular
            for targeted attacks.
        initialization_attack : :class:`Attack`
            Attack to use to find a starting point. Defaults to
            SaltAndPepperNoiseAttack.

        """

        self._starting_point = starting_point
        self._initialization_attack = initialization_attack
        yield from self.initialize_starting_point(a)

        if a.perturbed is None:
            warnings.warn(
                "Initialization failed. If the criterion is targeted,"
                " it might be necessary to pass an explicit starting"
                " point or targeted initialization attack."
            )
            return

        shape = a.unperturbed.shape
        N = a.unperturbed.size

        original = a.unperturbed.reshape(-1)
        x = a.perturbed.copy().reshape(-1)

        assert original.dtype == x.dtype

        while True:
            # draw random shuffling of all indices
            indices = list(range(N))
            rng.shuffle(indices)

            for index in indices:
                # change index
                old_value = x[index]
                new_value = original[index]
                if old_value == new_value:
                    continue
                x[index] = new_value

                # check if still adversarial
                _, is_adversarial = yield from a.forward_one(x.reshape(shape))

                # if adversarial, restart from there
                if is_adversarial:
                    logging.info(
                        "Reset value to original -> new distance:"
                        " {}".format(a.distance)
                    )
                    break

                # if not, undo change
                x[index] = old_value
            else:
                # no index was successful
                break

        logging.info("Starting binary searches")

        while True:
            # draw random shuffling of all indices
            indices = list(range(N))
            rng.shuffle(indices)

            # whether that run through all values made any improvement
            improved = False

            logging.info("Starting new loop through all values")

            for index in indices:
                # change index
                old_value = x[index]
                original_value = original[index]
                if old_value == original_value:
                    continue
                x[index] = original_value

                # check if still adversarial
                _, is_adversarial = yield from a.forward_one(x.reshape(shape))

                # if adversarial, no binary search needed
                if is_adversarial:  # pragma: no cover
                    logging.info(
                        "Reset value at {} to original ->"
                        " new distance: {}".format(index, a.distance)
                    )
                    improved = True
                else:
                    # binary search
                    adv_value = old_value
                    non_adv_value = original_value
                    best_adv_value = yield from self.binary_search(
                        a, x, index, adv_value, non_adv_value, shape
                    )

                    if old_value != best_adv_value:
                        x[index] = best_adv_value
                        improved = True
                        logging.info(
                            "Set value at {} from {} to {}"
                            " (original has {}) ->"
                            " new distance: {}".format(
                                index,
                                old_value,
                                best_adv_value,
                                original_value,
                                a.distance,
                            )
                        )

            if not improved:
                # no improvement for any of the indices
                break

    def binary_search(self, a, x, index, adv_value, non_adv_value, shape):
        for i in range(10):
            next_value = (adv_value + non_adv_value) / 2
            x[index] = next_value
            _, is_adversarial = yield from a.forward_one(x.reshape(shape))
            if is_adversarial:
                adv_value = next_value
            else:
                non_adv_value = next_value
        return adv_value

    def initialize_starting_point(self, a):
        starting_point = self._starting_point
        init_attack = self._initialization_attack

        if a.perturbed is not None:
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    "Ignoring starting_point because the attack"
                    " is applied to a previously found adversarial."
                )
            if init_attack is not None:  # pragma: no cover
                warnings.warn(
                    "Ignoring initialization_attack because the attack"
                    " is applied to a previously found adversarial."
                )
            return

        if starting_point is not None:
            yield from a.forward_one(starting_point)
            assert a.perturbed is not None, (
                "Invalid starting point provided. Please provide a starting "
                "point that is adversarial."
            )
            return

        if init_attack is None:
            init_attack = SaltAndPepperNoiseAttack
            logging.info(
                "Neither starting_point nor initialization_attack given."
                " Falling back to {} for initialization.".format(init_attack.__name__)
            )

        if issubclass(init_attack, Attack):
            # instantiate if necessary
            init_attack = init_attack()

        yield from init_attack.as_generator(a)
