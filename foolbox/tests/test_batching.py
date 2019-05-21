import numpy as np

from foolbox.batch_attacks import GradientAttack as Attack

from foolbox import run_parallel
from foolbox import run_sequential


def test_run_parallel(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(bn_images, bn_labels)

    advs2 = run_parallel(Attack, bn_model, bn_criterion, bn_images, bn_labels)
    advs2 = np.stack([a.perturbed for a in advs2])

    assert np.all(advs1 == advs2)


def test_run_sequential(bn_model, bn_criterion, bn_images, bn_labels):
    attack = Attack(bn_model, bn_criterion)
    advs1 = attack(bn_images, bn_labels)

    advs2 = run_sequential(Attack, bn_model, bn_criterion, bn_images, bn_labels)
    advs2 = np.stack([a.perturbed for a in advs2])

    assert np.all(advs1 == advs2)
