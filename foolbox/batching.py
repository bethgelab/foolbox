import logging
import numpy as np
import itertools
from .distances import MSE
from .yielding_adversarial import YieldingAdversarial


def run_sequential(create_attack_fn, model, criterion, inputs, labels,
                   distance=MSE, threshold=None, verbose=False, **kwargs):
    advs = [YieldingAdversarial(model, criterion, x, label,
                                distance=distance, threshold=threshold, verbose=verbose)
            for x, label in zip(inputs, labels)]
    attacks = [create_attack_fn().as_generator(adv, **kwargs) for adv in advs]

    supported_methods = {
        'forward_one': model.forward_one,
        'gradient_one': model.gradient_one,
        'backward_one': model.backward_one,
        'forward_and_gradient_one': model.forward_and_gradient_one,
    }

    for i, attack in enumerate(attacks):
        result = None
        while True:
            try:
                x = attack.send(result)
            except StopIteration:
                break
            method, args = x[0], x[1:]
            method = supported_methods[method]
            result = method(*args)
            assert result is not None
        logging.info('{} of {} attacks completed'.format(i + 1, len(advs)))
    return advs


def run_parallel(create_attack_fn, model, criterion, inputs, labels,
                 distance=MSE, threshold=None, verbose=False, **kwargs):
    advs = [YieldingAdversarial(model, criterion, x, label,
                                distance=distance, threshold=threshold, verbose=verbose)
            for x, label in zip(inputs, labels)]
    attacks = [create_attack_fn().as_generator(adv, **kwargs) for adv in advs]

    predictions = [None for _ in attacks]
    gradients = []
    backwards = []
    results = itertools.chain(predictions, gradients, backwards)

    while True:
        attacks_requesting_predictions = []
        predictions_args = []
        attacks_requesting_gradients = []
        gradients_args = []
        attacks_requesting_backwards = []
        backwards_args = []
        for attack, result in zip(attacks, results):
            try:
                x = attack.send(result)
            except StopIteration:
                continue
            method, args = x[0], x[1:]
            if method == 'forward_one':
                attacks_requesting_predictions.append(attack)
                predictions_args.append(args)
            elif method == 'gradient_one':
                attacks_requesting_gradients.append(attack)
                gradients_args.append(args)
            elif method == 'backward_one':
                attacks_requesting_backwards.append(attack)
                backwards_args.append(args)
            elif method == 'forward_and_gradient_one':
                raise NotImplementedError('batching support for forward_and_gradient_one'
                                          ' not yet implemented; please open an issue')
            else:
                assert False
        N_active_attacks = len(attacks_requesting_predictions) \
            + len(attacks_requesting_gradients) \
            + len(attacks_requesting_backwards)
        if N_active_attacks < len(predictions) + len(gradients) + len(backwards):  # noqa: E501
            # an attack completed in this iteration
            logging.info('{} of {} attacks completed'.format(len(advs) - N_active_attacks, len(advs)))  # noqa: E501
        if N_active_attacks == 0:
            break

        if len(attacks_requesting_predictions) > 0:
            logging.debug('calling forward with', len(attacks_requesting_predictions))  # noqa: E501
            predictions_args = map(np.stack, zip(*predictions_args))
            predictions = model.forward(*predictions_args)
        else:
            predictions = []

        if len(attacks_requesting_gradients) > 0:
            logging.debug('calling gradient with', len(attacks_requesting_gradients))  # noqa: E501
            gradients_args = map(np.stack, zip(*gradients_args))
            gradients = model.gradient(*gradients_args)
        else:
            gradients = []

        if len(attacks_requesting_backwards) > 0:
            logging.debug('calling backward with', len(attacks_requesting_backwards))  # noqa: E501
            backwards_args = map(np.stack, zip(*backwards_args))
            backwards = model.backward(*backwards_args)
        else:
            backwards = []

        attacks = itertools.chain(attacks_requesting_predictions, attacks_requesting_gradients, attacks_requesting_backwards)  # noqa: E501
        results = itertools.chain(predictions, gradients, backwards)
    return advs
