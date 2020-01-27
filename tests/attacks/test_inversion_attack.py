import foolbox.ext.native as fbn


def test_inversion_attack(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    attack = fbn.attacks.InversionAttack(fmodel)
    advs = attack(x, y)
    assert advs.shape == x.shape
    y_advs = fmodel.forward(advs).argmax(axis=-1)
    assert (y_advs != y).any()
