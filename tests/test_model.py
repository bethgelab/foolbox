import eagerpy as ep


def test_bounds(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    min_, max_ = fmodel.bounds()
    assert min_ < max_
    assert (x >= min_).all()
    assert (x <= max_).all()


def test_forward(fmodel_and_data):
    fmodel, x, y = fmodel_and_data
    logits = ep.astensor(fmodel.forward(x.tensor))
    assert logits.ndim == 2
    assert len(logits) == len(x) == len(y)
    _, num_classes = logits.shape
    assert (y >= 0).all()
    assert (y < num_classes).all()
