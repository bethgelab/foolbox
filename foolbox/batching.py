import logging
import numpy as np
import itertools
from .distances import MSE
from .yielding_adversarial import YieldingAdversarial


def run_sequential(create_attack_fn, model, criterion, inputs, labels,
                   distance=MSE, threshold=None, verbose=False, **kwargs):
    """
    Runs the same type of attack vor multiple inputs sequentially without
    batching them.

    Parameters
    ----------
    create_attack_fn : a function returning an :class:`Attack` instance
        The attack to use.
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` class or list of :class:`Criterion` classes
        The criterion/criteria that determine(s) which inputs are adversarial.
    inputs :  a :class:`numpy.ndarray`
        The unperturbed inputs to which the adversarial input should be as close
        as possible.
    labels :  a :class:`numpy.ndarray`
        The ground-truth labels of the unperturbed inputs.
    distance : a :class:`Distance` class
        The measure used to quantify how close inputs are.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the :class:`Adversarial`.`reached_threshold()` method can
         be used to check
        if the threshold has been reached.
    verbose : bool
        Whether the adversarial examples should be created in verbose mode.
    kwargs : dict
         The optional keywords passed to create_attack_fn.

    Returns
    -------
    The list of generated adversarial examples.
    """

    assert len(inputs) == len(labels), 'The number of inputs must match the number of labels.'  # noqa: E501

    # if only one criterion has been passed use the same one for all inputs
    if not isinstance(criterion, (list, tuple)):
        criterion = [criterion] * len(inputs)
    else:
        assert len(criterion) == len(inputs), 'The number of criteria must match the number of inputs.'  # noqa: E501

    # if only one distance has been passed use the same one for all inputs
    if not isinstance(distance, (list, tuple)):
        distance = [distance] * len(inputs)
    else:
        assert len(distance) == len(inputs), 'The number of distances must match the number of inputs.'  # noqa: E501

    advs = [YieldingAdversarial(model, _criterion, x, label,
                                distance=_distance, threshold=threshold,
                                verbose=verbose)
            for _criterion, _distance, x, label in zip(criterion, distance,
                                                       inputs, labels)]
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
    """
        Runs the same type of attack vor multiple inputs in parallel by
        batching them.

        Parameters
        ----------
        create_attack_fn : a function returning an :class:`Attack` instance
            The attack to use.
        model : a :class:`Model` instance
            The model that should be fooled by the adversarial.
        criterion : a :class:`Criterion` class or list of :class:`Criterion` classes
            The criterion/criteria that determine(s) which inputs are adversarial.
        inputs :  a :class:`numpy.ndarray`
            The unperturbed inputs to which the adversarial input should be as close
            as possible.
        labels :  a :class:`numpy.ndarray`
            The ground-truth labels of the unperturbed inputs.
        distance : a :class:`Distance` class or list of :class:`Distance` classes
            The measure(s) used to quantify how close inputs are.
        threshold : float or :class:`Distance`
            If not None, the attack will stop as soon as the adversarial
            perturbation has a size smaller than this threshold. Can be
            an instance of the :class:`Distance` class passed to the distance
            argument, or a float assumed to have the same unit as the
            the given distance. If None, the attack will simply minimize
            the distance as good as possible. Note that the threshold only
            influences early stopping of the attack; the returned adversarial
            does not necessarily have smaller perturbation size than this
            threshold; the :class:`Adversarial`.`reached_threshold()` method can
             be used to check
            if the threshold has been reached.
        verbose : bool
            Whether the adversarial examples should be created in verbose mode.
        kwargs : dict
             The optional keywords passed to create_attack_fn.

        Returns
        -------
        The list of generated adversarial examples.
        """

    assert len(inputs) == len(labels), 'The number of inputs must match the number of labels.'  # noqa: E501

    # if only one criterion has been passed use the same one for all inputs
    if not isinstance(criterion, (list, tuple)):
        criterion = [criterion] * len(inputs)
    else:
        assert len(criterion) == len(inputs), 'The number of criteria must match the number of inputs.'  # noqa: E501

    # if only one distance has been passed use the same one for all inputs
    if not isinstance(distance, (list, tuple)):
        distance = [distance] * len(inputs)
    else:
        assert len(distance) == len(inputs), 'The number of distances must match the number of inputs.'  # noqa: E501

    advs = [YieldingAdversarial(model, _criterion, x, label,
                                distance=_distance, threshold=threshold,
                                verbose=verbose)
            for _criterion, _distance, x, label in zip(criterion, distance,
                                                       inputs, labels)]
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
                raise NotImplementedError('batching support for forward_and_'
                                          'gradient_one not yet implemented; '
                                          'please open an issue')
            else:
                assert False
        n_active_attacks = len(attacks_requesting_predictions) \
            + len(attacks_requesting_gradients) \
            + len(attacks_requesting_backwards)
        if n_active_attacks < len(predictions) + len(gradients) + len(backwards):  # noqa: E501
            # an attack completed in this iteration
            logging.info('{} of {} attacks completed'.format(len(advs) - n_active_attacks, len(advs)))  # noqa: E501
        if n_active_attacks == 0:
            break

        if len(attacks_requesting_predictions) > 0:
            logging.debug('calling forward with {}'.format(len(attacks_requesting_predictions)))  # noqa: E501
            predictions_args = map(np.stack, zip(*predictions_args))
            predictions = model.forward(*predictions_args)
        else:
            predictions = []

        if len(attacks_requesting_gradients) > 0:
            logging.debug('calling gradient with {}'.format(len(attacks_requesting_gradients)))  # noqa: E501
            gradients_args = map(np.stack, zip(*gradients_args))
            gradients = model.gradient(*gradients_args)
        else:
            gradients = []

        if len(attacks_requesting_backwards) > 0:
            logging.debug('calling backward with {}'.format(len(attacks_requesting_backwards)))  # noqa: E501
            backwards_args = map(np.stack, zip(*backwards_args))
            backwards = model.backward(*backwards_args)
        else:
            backwards = []

        attacks = itertools.chain(attacks_requesting_predictions, attacks_requesting_gradients, attacks_requesting_backwards)  # noqa: E501
        results = itertools.chain(predictions, gradients, backwards)
    return advs
