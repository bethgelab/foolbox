import pytest
import numpy as np
from foolbox import criteria


def test_abstract_criterion():
    with pytest.raises(TypeError):
        criteria.Criterion()


def test_base_criterion():

    class TestCriterion(criteria.Criterion):

        def is_adversarial(self, predictions, label):
            return False

    criterion = TestCriterion()
    assert criterion.name() == 'TestCriterion'


def test_combined_criteria():
    c1 = criteria.Misclassification()
    c2 = criteria.OriginalClassProbability(0.2)
    c3 = c1 & c2

    probabilities = np.array([0.09, 0.11, 0.39, 0.41])
    predictions = np.log(probabilities)

    for i in range(len(predictions)):
        b1 = c1.is_adversarial(predictions, i)
        b2 = c2.is_adversarial(predictions, i)
        b3 = c3.is_adversarial(predictions, i)

        assert (b1 and b2) == b3

    assert c1.name() == 'Top1Misclassification'
    assert c2.name() == 'OriginalClassProbability-0.2000'
    assert c3.name() == c2.name() + '__' + c1.name()


def test_misclassfication():
    c = criteria.Misclassification()
    predictions = np.array([0.1, 0.5, 0.7, 0.4])
    assert c.is_adversarial(predictions, 0)
    assert c.is_adversarial(predictions, 1)
    assert not c.is_adversarial(predictions, 2)
    assert c.is_adversarial(predictions, 3)


def test_misclassification_names():
    c = criteria.Misclassification()
    c1 = criteria.TopKMisclassification(k=1)
    c5 = criteria.TopKMisclassification(k=5)
    assert c.name() == c1.name()
    assert c1.name() != c5.name()
    c22 = criteria.TopKMisclassification(k=22)
    assert '22' in c22.name()


def test_top_k_misclassfication():
    predictions = np.array([0.1, 0.5, 0.7, 0.4])

    c = criteria.TopKMisclassification(k=1)
    assert c.is_adversarial(predictions, 0)
    assert c.is_adversarial(predictions, 1)
    assert not c.is_adversarial(predictions, 2)
    assert c.is_adversarial(predictions, 3)

    c = criteria.TopKMisclassification(k=2)
    assert c.is_adversarial(predictions, 0)
    assert not c.is_adversarial(predictions, 1)
    assert not c.is_adversarial(predictions, 2)
    assert c.is_adversarial(predictions, 3)


def test_target_class():
    predictions = np.array([0.1, 0.5, 0.7, 0.4])

    c = criteria.TargetClass(3)
    for i in range(len(predictions)):
        assert not c.is_adversarial(predictions, i)

    assert c.name() == 'TargetClass-3'

    c = criteria.TargetClass(2)
    for i in range(len(predictions)):
        assert c.is_adversarial(predictions, i)

    assert c.name() == 'TargetClass-2'


def test_original_class_probability():
    predictions = np.array([0.1, 0.5, 0.7, 10., 0.4])

    c = criteria.OriginalClassProbability(0.1)
    assert c.is_adversarial(predictions, 0)
    assert c.is_adversarial(predictions, 1)
    assert c.is_adversarial(predictions, 2)
    assert not c.is_adversarial(predictions, 3)
    assert c.is_adversarial(predictions, 4)

    assert '0.1' in c.name()


def test_target_class_probability():
    predictions = np.array([0.1, 0.5, 0.7, 10., 0.4])

    for t in [0, 1, 2, 4]:
        c = criteria.TargetClassProbability(0, p=0.9)
        for i in range(len(predictions)):
            assert not c.is_adversarial(predictions, i)

    c = criteria.TargetClassProbability(3, p=0.9)
    for i in range(len(predictions)):
        assert c.is_adversarial(predictions, i)

    assert '3' in c.name()
    assert '0.9' in c.name()


def test_confident_misclassification():
    predictions = np.array([0.1, 0.5, 0.7, 10., 0.4])  # 99%

    for p in [0.1, 0.5, 0.9]:
        c = criteria.ConfidentMisclassification(p=p)
        for i in [0, 1, 2, 4]:
            assert c.is_adversarial(predictions, i)
        assert not c.is_adversarial(predictions, 3)

    predictions = np.array([0.1, 0.5, 0.7, 10., 10.1])  # 47% and 52%

    for p in [0.1, 0.5, 0.9]:
        c = criteria.ConfidentMisclassification(p=p)
        for i in range(4):
            expect = i < 4 and p <= 0.5
            assert c.is_adversarial(predictions, i) == expect

    c = criteria.ConfidentMisclassification(p=0.9)
    assert '0.9' in c.name()
