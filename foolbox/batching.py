import logging
import numpy as np
import itertools
from .distances import MSE
from .yielding_adversarial import YieldingAdversarial


def run_serial_attack(create_attack_fn, model, criterion, inputs, labels,
                      distance=MSE, threshold=None, verbose=False):
    advs = [YieldingAdversarial(model, criterion, x, label,
                                distance=distance, threshold=threshold, verbose=verbose)  # noqa: E501
            for x, label in zip(inputs, labels)]
    attacks = [create_attack_fn()(adv) for adv in advs]

    for i, attack in enumerate(attacks):
        result = None
        while True:
            try:
                x = attack.send(result)
            except StopIteration:
                break
            method, args = x[0], x[1:]
            method = {
                'predictions': model.predictions,
                'gradient': model.gradient,
                'backward': model.backward,
            }[method]
            result = method(*args)
            assert result is not None
        logging.info('{} of {} attacks completed'.format(i + 1, len(advs)))
    return advs


def run_parallel_attack(create_attack_fn, model, criterion, inputs, labels,
                        distance=MSE, threshold=None, verbose=False):
    advs = [YieldingAdversarial(model, criterion, x, label,
                                distance=distance, threshold=threshold, verbose=verbose)  # noqa: E501
            for x, label in zip(inputs, labels)]
    attacks = [create_attack_fn()(adv) for adv in advs]

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
            if method == 'predictions':
                attacks_requesting_predictions.append(attack)
                predictions_args.append(args)
            elif method == 'gradient':
                attacks_requesting_gradients.append(attack)
                gradients_args.append(args)
            elif method == 'backward':
                attacks_requesting_backwards.append(attack)
                backwards_args.append(args)
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
            logging.debug('calling batch_predictions with', len(attacks_requesting_predictions))  # noqa: E501
            predictions_args = map(np.stack, zip(*predictions_args))
            predictions = model.batch_predictions(*predictions_args)
        else:
            predictions = []

        if len(attacks_requesting_gradients) > 0:
            logging.debug('calling batch_gradients with', len(attacks_requesting_gradients))  # noqa: E501
            gradients_args = map(np.stack, zip(*gradients_args))
            gradients = model.batch_gradients(*gradients_args)
        else:
            gradients = []

        if len(attacks_requesting_backwards) > 0:
            logging.debug('calling batch_backward with', len(attacks_requesting_backwards))  # noqa: E501
            backwards_args = map(np.stack, zip(*backwards_args))
            backwards = model.batch_backward(*backwards_args)
        else:
            backwards = []

        attacks = itertools.chain(attacks_requesting_predictions, attacks_requesting_gradients, attacks_requesting_backwards)  # noqa: E501
        results = itertools.chain(predictions, gradients, backwards)
    return advs
