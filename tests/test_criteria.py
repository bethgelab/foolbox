import eagerpy as ep
import foolbox.ext.native as fbn


def test_correct_unperturbed(fmodel_and_data):
    fmodel, inputs, _ = fmodel_and_data
    perturbed = inputs
    logits = fmodel.forward(perturbed)
    labels = logits.argmax(axis=-1)

    is_adv = fbn.Misclassification()(inputs, labels, perturbed, logits)
    assert not is_adv.any()

    is_adv = fbn.misclassification(inputs, labels, perturbed, logits)
    assert not is_adv.any()

    _, num_classes = logits.shape
    target_classes = (labels + 1) % num_classes
    is_adv = fbn.TargetedMisclassification(target_classes)(
        inputs, labels, perturbed, logits
    )
    assert not is_adv.any()

    is_adv = (fbn.misclassification & fbn.misclassification)(
        inputs, labels, perturbed, logits
    )
    assert not is_adv.any()


def test_wrong_unperturbed(fmodel_and_data):
    fmodel, inputs, _ = fmodel_and_data
    perturbed = inputs
    logits = fmodel.forward(perturbed)
    _, num_classes = logits.shape
    labels = logits.argmax(axis=-1)
    labels = (labels + 1) % num_classes

    is_adv = fbn.Misclassification()(inputs, labels, perturbed, logits)
    assert is_adv.all()

    is_adv = fbn.misclassification(inputs, labels, perturbed, logits)
    assert is_adv.all()

    target_classes = (labels + 1) % num_classes
    is_adv = fbn.TargetedMisclassification(target_classes)(
        inputs, labels, perturbed, logits
    )
    if num_classes > 2:
        assert not is_adv.any()
    else:
        assert is_adv.all()

    is_adv = (fbn.misclassification & fbn.misclassification)(
        inputs, labels, perturbed, logits
    )
    assert is_adv.all()

    is_adv = (
        fbn.TargetedMisclassification(labels)
        & fbn.TargetedMisclassification(target_classes)
    )(inputs, labels, perturbed, logits)
    assert not is_adv.any()


def test_repr_object():
    assert repr(object()).startswith("<")


def test_repr_misclassification():
    assert not repr(fbn.misclassification).startswith("<")


def test_repr_and():
    assert not repr(fbn.misclassification & fbn.misclassification).startswith("<")


def test_repr_targeted_misclassification(dummy):
    target_classes = ep.arange(dummy, 10)
    assert not repr(fbn.TargetedMisclassification(target_classes)).startswith("<")
