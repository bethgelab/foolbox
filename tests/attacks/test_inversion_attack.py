import foolbox.ext.native as fbn


def test_inversion_attack(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    attack = fbn.attacks.InversionAttack(fmodel)
    advs = attack(x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)
