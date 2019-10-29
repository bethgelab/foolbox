import logging
import numpy as np
import itertools
from .distances import MSE
from .adversarial import Adversarial


def run_sequential(
    create_attack_fn,
    model,
    criterion,
    inputs,
    labels,
    distance=MSE,
    threshold=None,
    verbose=False,
    individual_kwargs=None,
    **kwargs
):
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
    individual_kwargs : list of dict
         The optional keywords passed to create_attack_fn that should be
         different for each of the input samples. For each input a different
         set of arguments will be used.
    kwargs : dict
        The optional keywords passed to create_attack_fn that are common for
        every element in the batch.

    Returns
    -------
    The list of generated adversarial examples.
    """

    assert len(inputs) == len(
        labels
    ), "The number of inputs must match the number of labels."  # noqa: E501

    # if only one criterion has been passed use the same one for all inputs
    if not isinstance(criterion, (list, tuple)):
        criterion = [criterion] * len(inputs)
    else:
        assert len(criterion) == len(
            inputs
        ), "The number of criteria must match the number of inputs."  # noqa: E501

    # if only one distance has been passed use the same one for all inputs
    if not isinstance(distance, (list, tuple)):
        distance = [distance] * len(inputs)
    else:
        assert len(distance) == len(
            inputs
        ), "The number of distances must match the number of inputs."  # noqa: E501

    if individual_kwargs is None:
        individual_kwargs = [kwargs] * len(inputs)
    else:
        assert isinstance(
            individual_kwargs, (list, tuple)
        ), "Individual_kwargs must be a list or None."  # noqa: E501
        assert len(individual_kwargs) == len(
            inputs
        ), "The number of individual_kwargs must match the number of inputs."  # noqa: E501

        for i in range(len(individual_kwargs)):
            assert isinstance(individual_kwargs[i], dict)
            individual_kwargs[i] = {**kwargs, **individual_kwargs[i]}

    advs = [
        Adversarial(
            model,
            _criterion,
            x,
            label,
            distance=_distance,
            threshold=threshold,
            verbose=verbose,
        )
        for _criterion, _distance, x, label in zip(criterion, distance, inputs, labels)
    ]
    attacks = [
        create_attack_fn().as_generator(adv, **kwargs)
        for adv, kwargs in zip(advs, individual_kwargs)
    ]

    supported_methods = {
        "forward_one": model.forward_one,
        "gradient_one": model.gradient_one,
        "backward_one": model.backward_one,
        "forward_and_gradient_one": model.forward_and_gradient_one,
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
        logging.info("{} of {} attacks completed".format(i + 1, len(advs)))
    return advs


def run_parallel(  # noqa: C901
    create_attack_fn,
    model,
    criterion,
    inputs,
    labels,
    distance=MSE,
    threshold=None,
    verbose=False,
    individual_kwargs=None,
    **kwargs
):
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
    individual_kwargs : list of dict
         The optional keywords passed to create_attack_fn that should be
         different for each of the input samples. For each input a different
         set of arguments will be used.
    kwargs : dict
        The optional keywords passed to create_attack_fn that are common for
        every element in the batch.

    Returns
    -------
    The list of generated adversarial examples.
    """

    assert len(inputs) == len(
        labels
    ), "The number of inputs must match the number of labels."  # noqa: E501

    # if only one criterion has been passed use the same one for all inputs
    if not isinstance(criterion, (list, tuple)):
        criterion = [criterion] * len(inputs)
    else:
        assert len(criterion) == len(
            inputs
        ), "The number of criteria must match the number of inputs."  # noqa: E501

    # if only one distance has been passed use the same one for all inputs
    if not isinstance(distance, (list, tuple)):
        distance = [distance] * len(inputs)
    else:
        assert len(distance) == len(
            inputs
        ), "The number of distances must match the number of inputs."  # noqa: E501

    if individual_kwargs is None:
        individual_kwargs = [kwargs] * len(inputs)
    else:
        assert isinstance(
            individual_kwargs, (list, tuple)
        ), "Individual_kwargs must be a list or None."  # noqa: E501
        assert len(individual_kwargs) == len(
            inputs
        ), "The number of individual_kwargs must match the number of inputs."  # noqa: E501

        for i in range(len(individual_kwargs)):
            assert isinstance(individual_kwargs[i], dict)
            individual_kwargs[i] = {**kwargs, **individual_kwargs[i]}

    advs = [
        Adversarial(
            model,
            _criterion,
            x,
            label,
            distance=_distance,
            threshold=threshold,
            verbose=verbose,
        )
        for _criterion, _distance, x, label in zip(criterion, distance, inputs, labels)
    ]
    attacks = [
        create_attack_fn().as_generator(adv, **kwargs)
        for adv, kwargs in zip(advs, individual_kwargs)
    ]

    predictions = [None for _ in attacks]
    gradients = []
    backwards = []
    prediction_gradients = []

    batched_predictions = []
    results = itertools.chain(
        predictions, gradients, backwards, prediction_gradients, batched_predictions
    )

    while True:
        attacks_requesting_predictions = []
        predictions_args = []
        attacks_requesting_gradients = []
        gradients_args = []
        attacks_requesting_backwards = []
        backwards_args = []
        attacks_requesting_prediction_gradients = []
        predictions_gradients_args = []
        attacks_requesting_batched_predictions = []
        batched_predictions_args = []
        for attack, result in zip(attacks, results):
            try:
                x = attack.send(result)
            except StopIteration:
                continue
            method, args = x[0], x[1:]

            if method == "forward_one":
                attacks_requesting_predictions.append(attack)
                predictions_args.append(args)
            elif method == "gradient_one":
                attacks_requesting_gradients.append(attack)
                gradients_args.append(args)
            elif method == "backward_one":
                attacks_requesting_backwards.append(attack)
                backwards_args.append(args)
            elif method == "forward_and_gradient_one":
                attacks_requesting_prediction_gradients.append(attack)
                predictions_gradients_args.append(args)
            elif method == "forward":
                attacks_requesting_batched_predictions.append(attack)
                batched_predictions_args.append(args)
            else:
                assert False
        n_active_attacks = (
            len(attacks_requesting_predictions)
            + len(attacks_requesting_gradients)
            + len(attacks_requesting_backwards)
            + len(attacks_requesting_prediction_gradients)
            + len(attacks_requesting_batched_predictions)
        )
        if n_active_attacks < len(predictions) + len(gradients) + len(backwards) + len(
            prediction_gradients
        ) + len(
            batched_predictions
        ):  # noqa: E501
            # an attack completed in this iteration
            logging.info(
                "{} of {} attacks completed".format(
                    len(advs) - n_active_attacks, len(advs)
                )
            )  # noqa: E501
        if n_active_attacks == 0:
            break

        if len(attacks_requesting_predictions) > 0:
            logging.debug(
                "calling forward with {}".format(len(attacks_requesting_predictions))
            )  # noqa: E501
            predictions_args = map(np.stack, zip(*predictions_args))
            predictions = model.forward(*predictions_args)
        else:
            predictions = []

        if len(attacks_requesting_batched_predictions) > 0:
            logging.debug(
                "calling native forward with {}".format(
                    len(attacks_requesting_batched_predictions)
                )
            )  # noqa: E501

            # we are only interested in the first argument
            inputs = [x[0] for x in batched_predictions_args]

            # merge individual batches into one larger super-batch
            batch_lengths = [len(x) for x in inputs]
            batch_splits = np.cumsum(batch_lengths)
            inputs = np.concatenate([x for x in inputs])

            # split super-batch back into individual batches
            batched_predictions = model.forward(inputs)
            batched_predictions = np.split(batched_predictions, batch_splits, axis=0)

        else:
            batched_predictions = []

        if len(attacks_requesting_gradients) > 0:
            logging.debug(
                "calling gradient with {}".format(len(attacks_requesting_gradients))
            )  # noqa: E501
            gradients_args = map(np.stack, zip(*gradients_args))
            gradients = model.gradient(*gradients_args)
        else:
            gradients = []

        if len(attacks_requesting_backwards) > 0:
            logging.debug(
                "calling backward with {}".format(len(attacks_requesting_backwards))
            )  # noqa: E501
            backwards_args = map(np.stack, zip(*backwards_args))
            backwards = model.backward(*backwards_args)
        else:
            backwards = []
        if len(attacks_requesting_prediction_gradients) > 0:
            logging.debug(
                "calling forward_and_gradient_one with {}".format(
                    len(attacks_requesting_prediction_gradients)
                )
            )  # noqa: E501

            predictions_gradients_args = map(np.stack, zip(*predictions_gradients_args))

            prediction_gradients = model.forward_and_gradient(
                *predictions_gradients_args
            )

            prediction_gradients = list(zip(*prediction_gradients))
        else:
            prediction_gradients = []

        attacks = itertools.chain(
            attacks_requesting_predictions,
            attacks_requesting_gradients,
            attacks_requesting_backwards,
            attacks_requesting_prediction_gradients,
            attacks_requesting_batched_predictions,
        )
        results = itertools.chain(
            predictions, gradients, backwards, prediction_gradients, batched_predictions
        )
    return advs
