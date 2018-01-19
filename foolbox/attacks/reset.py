import warnings
import random
import logging

from .base import Attack
from .blended_noise import BlendedUniformNoiseAttack


class ResetAttack(Attack):
    """Starts with an adversarial and resets as many values as possible
    to the values of the original.

    Based on the FindWitness algorithm described in [1].

    References
    ----------
    .. [1] Daniel Lowd, Christopher Meek, "Adversarial Learning",
           Proceedings of the Eleventh ACM SIGKDD International Conference
           on Knowledge Discovery in Data Mining, KDD 2005.

    """

    def _apply(self, a, starting_point=None, initialization_attack=None):
        self._starting_point = starting_point
        self._initialization_attack = initialization_attack
        self.initialize_starting_point(a)

        if a.image is None:
            warnings.warn(
                'Initialization failed. If the criterion is targeted,'
                ' it might be necessary to pass an explicit starting'
                ' point or targeted initialization attack.')
            return

        shape = a.original_image.shape
        N = a.original_image.size

        original = a.original_image.reshape(-1)
        x = a.image.reshape(-1)

        assert original.dtype == x.dtype

        while True:
            # draw random shuffling of all indices
            indices = list(range(N))
            random.shuffle(indices)

            for index in indices:
                # change index
                old_value = x[index]
                new_value = original[index]
                if old_value == new_value:
                    continue
                x[index] = new_value

                # check if still adversarial
                _, is_adversarial = a.predictions(x.reshape(shape))

                # if adversarial, restart from there
                if is_adversarial:
                    logging.info('Reduced distance: {}'.format(a.distance))
                    break

                # if not, undo change
                x[index] = old_value
            else:
                # no index was succesful
                break

    def initialize_starting_point(self, a):
        starting_point = self._starting_point
        init_attack = self._initialization_attack

        if a.image is not None:
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring starting_point because the attack'
                    ' is applied to a previously found adversarial.')
            if init_attack is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring initialization_attack because the attack'
                    ' is applied to a previously found adversarial.')
            return

        if starting_point is not None:
            a.predictions(starting_point)
            assert a.image is not None, ('Invalid starting point provided.'
                                         ' Please provide a starting point'
                                         ' that is adversarial.')
            return

        if init_attack is None:
            init_attack = BlendedUniformNoiseAttack
            print(
                'Neither starting_point nor initialization_attack given.'
                ' Falling back to {} for initialization.'.format(
                    init_attack.__name__))

        if issubclass(init_attack, Attack):
            # instantiate if necessary
            init_attack = init_attack()

        init_attack(a)
