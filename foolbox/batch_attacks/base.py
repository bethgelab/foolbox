import warnings
import logging
import functools
import numpy as np

from ..attacks.base import Attack
from ..yielding_adversarial import YieldingAdversarial
from ..adversarial import StopAttack
from ..batching import run_parallel


class BatchAttack(Attack):
    def __call__(self, inputs, labels, unpack=True, **kwargs):
        assert isinstance(inputs, np.ndarray)
        assert isinstance(labels, np.ndarray)

        if len(inputs) != len(labels):
            raise ValueError('The number of inputs and labels needs to be equal')

        model = self._default_model
        criterion = self._default_criterion
        distance = self._default_distance
        threshold = self._default_threshold

        if model is None:
            raise ValueError('The attack needs to be initialized with a model')
        if criterion is None:
            raise ValueError('The attack needs to be initialized with a criterion')
        if distance is None:
            raise ValueError('The attack needs to be initialized with a distance')

        create_attack_fn = self.__class__
        advs = run_parallel(create_attack_fn, model, criterion, inputs, labels,
                            distance=distance, threshold=threshold, **kwargs)

        if unpack:
            advs = [a.perturbed for a in advs]
            advs = [p if p is not None else np.full_like(u, np.nan) for p, u in zip(advs, inputs)]
            advs = np.stack(advs)
        return advs


def generator_decorator(generator):
    @functools.wraps(generator)
    def wrapper(self, a, **kwargs):
        assert isinstance(a, YieldingAdversarial)

        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original input'
                          ' is already misclassified and the adversarial thus'
                          ' has a distance of 0.')
        elif a.reached_threshold():
            warnings.warn('Not running the attack because the given treshold'
                          ' is already reached')
        else:
            try:
                _ = yield from generator(self, a, **kwargs)
                assert _ is None, 'decorated __call__ method must return None'
            except StopAttack:
                # if a threshold is specified, StopAttack will be thrown
                # when the treshold is reached; thus we can do early
                # stopping of the attack
                logging.info('threshold reached, stopping attack')

        if a.perturbed is None:
            warnings.warn('{} did not find an adversarial, maybe the model'
                          ' or the criterion is not supported by this'
                          ' attack.'.format(self.name()))
        return a

    return wrapper
