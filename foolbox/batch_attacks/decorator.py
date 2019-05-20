import warnings
import logging
import functools

from ..adversarial import Adversarial
from ..yielding_adversarial import YieldingAdversarial
from ..adversarial import StopAttack


def generator_call_decorator(call_fn):
    @functools.wraps(call_fn)
    def wrapper(self, input_or_adv, label=None, unpack=True, **kwargs):
        assert input_or_adv is not None

        if isinstance(input_or_adv, YieldingAdversarial):
            a = input_or_adv
            if label is not None:
                raise ValueError('Label must not be passed when input_or_adv'
                                 ' is an Adversarial instance')
        elif isinstance(input_or_adv, Adversarial):
            raise ValueError('If you pass an Adversarial instance, it must be a YieldingAdversarial instance'
                             ' when calling a batch attack like this one.')
        else:
            if label is None:
                raise ValueError('Label must be passed when input_or_adv is'
                                 ' not an Adversarial instance')
            else:
                model = self._default_model
                criterion = self._default_criterion
                distance = self._default_distance
                threshold = self._default_threshold
                if model is None or criterion is None:
                    raise ValueError('The attack needs to be initialized'
                                     ' with a model and a criterion or it'
                                     ' needs to be called with an Adversarial'
                                     ' instance.')
                a = Adversarial(model, criterion, input_or_adv, label,
                                distance=distance, threshold=threshold)

        assert a is not None

        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original input'
                          ' is already misclassified and the adversarial thus'
                          ' has a distance of 0.')
        elif a.reached_threshold():
            warnings.warn('Not running the attack because the given treshold'
                          ' is already reached')
        else:
            try:
                _ = yield from call_fn(self, a, label=None, unpack=None, **kwargs)
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

        if unpack:
            return a.perturbed
        else:
            return a

    return wrapper
