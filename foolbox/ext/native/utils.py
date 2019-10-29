import eagerpy as ep


def accuracy(fmodel, images, labels):
    logits = ep.astensor(fmodel.forward(images))
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).float32().mean()
    return accuracy.item()
