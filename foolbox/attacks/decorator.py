import warnings
import functools

from ..adversarial import Adversarial


def call_decorator(call_fn):
    @functools.wraps(call_fn)
    def wrapper(
            self,
            a,
            label=None,
            unpack=True,
            **kwargs):

        assert a is not None

        if isinstance(a, Adversarial):
            if label is not None:
                raise ValueError('Label must not be passed when image is'
                                 ' an Adversarial instance')
        else:
            if label is None:
                raise ValueError('Label must be passed when image is'
                                 ' not an Adversarial instance')
            else:
                model = self._default_model
                criterion = self._default_criterion
                if model is None or criterion is None:
                    raise ValueError('The attack needs to be initialized'
                                     ' with a model and a criterion or it'
                                     ' needs to be called with an Adversarial'
                                     ' instance.')
                a = Adversarial(model, criterion, a, label)

        assert a is not None

        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original image'
                          ' is already misclassified and the adversarial thus'
                          ' has a distance of 0.')
        else:
            _ = call_fn(self, a, **kwargs)
            assert _ is None, 'decorated __call__ method must return None'

        if a.image is None:
            warnings.warn('{} did not find an adversarial, maybe the model'
                          ' or the criterion is not supported by this'
                          ' attack.'.format(self.name()))

        if unpack:
            return a.image
        else:
            return a

    return wrapper
