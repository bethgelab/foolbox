import foolbox.ext.native as fbn


def test_accracy(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    accuracy = fbn.accuracy(fmodel, x, y)
    assert 0 <= accuracy <= 1
    assert accuracy > 0.5
    y = fmodel.forward(x).argmax(axis=-1)
    accuracy = fbn.accuracy(fmodel, x, y)
    assert accuracy == 1
