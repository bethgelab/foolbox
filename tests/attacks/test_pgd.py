import eagerpy as ep
from foolbox.ext.native.attacks import PGD


def test_linf_pgd(fmodel_and_data):
    fmodel, x, y, batch_size, num_classes = fmodel_and_data
    y = ep.astensor(fmodel.forward(x)).argmax(axis=-1)

    attack = PGD(fmodel)
    advs = attack(x, y, rescale=False, epsilon=0.3)

    perturbation = ep.astensor(advs - x)
    y_advs = ep.astensor(fmodel.forward(advs)).argmax(axis=-1)

    assert x.shape == advs.shape
    assert perturbation.abs().max() <= 0.3 + 1e-7
    assert (y_advs == y).float32().mean() < 1
