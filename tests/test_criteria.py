from typing import Tuple
import foolbox as fbn
import eagerpy as ep


def test_correct_unperturbed(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, inputs, _ = fmodel_and_data
    perturbed = inputs
    logits = fmodel(perturbed)
    labels = logits.argmax(axis=-1)

    is_adv = fbn.Misclassification(labels)(perturbed, logits)
    assert not is_adv.any()

    _, num_classes = logits.shape
    target_classes = (labels + 1) % num_classes
    is_adv = fbn.TargetedMisclassification(target_classes)(perturbed, logits)
    assert not is_adv.any()

    combined = fbn.Misclassification(labels) & fbn.Misclassification(labels)
    is_adv = combined(perturbed, logits)
    assert not is_adv.any()


def test_wrong_unperturbed(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor]
) -> None:
    fmodel, inputs, _ = fmodel_and_data
    perturbed = inputs
    logits = fmodel(perturbed)
    _, num_classes = logits.shape
    labels = logits.argmax(axis=-1)
    labels = (labels + 1) % num_classes

    is_adv = fbn.Misclassification(labels)(perturbed, logits)
    assert is_adv.all()

    target_classes = (labels + 1) % num_classes
    is_adv = fbn.TargetedMisclassification(target_classes)(perturbed, logits)
    if num_classes > 2:
        assert not is_adv.any()
    else:
        assert is_adv.all()

    is_adv = (fbn.Misclassification(labels) & fbn.Misclassification(labels))(
        perturbed, logits
    )
    assert is_adv.all()

    combined = fbn.TargetedMisclassification(labels) & fbn.TargetedMisclassification(
        target_classes
    )
    is_adv = combined(perturbed, logits)
    assert not is_adv.any()


def test_repr_object() -> None:
    assert repr(object()).startswith("<")


def test_repr_misclassification(dummy: ep.Tensor) -> None:
    labels = ep.arange(dummy, 10)
    assert not repr(fbn.Misclassification(labels)).startswith("<")


def test_repr_and(dummy: ep.Tensor) -> None:
    labels = ep.arange(dummy, 10)
    assert not repr(
        fbn.Misclassification(labels) & fbn.Misclassification(labels)
    ).startswith("<")


def test_repr_targeted_misclassification(dummy: ep.Tensor) -> None:
    target_classes = ep.arange(dummy, 10)
    assert not repr(fbn.TargetedMisclassification(target_classes)).startswith("<")
