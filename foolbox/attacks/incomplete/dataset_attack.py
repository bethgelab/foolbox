import eagerpy as ep
import numpy as np

from .base import MinimizationAttack
from ..devutils import wrap
from ..models.base import Model


# TODO: rewrite this to make it more efficient and to allow both storing on GPU and CPU


class DatasetAttack(MinimizationAttack):
    """This is a helper attack that makes it straight-forward to choose initialisation points
       for boundary-type attacks from a given data set. All it does is to store samples and
       the predicted responses of a given model in order to select suitable adversarial images
       from the given data set.
    """

    def __init__(self):
        self.samples = []
        self.labels = []

    def feed(self, model: Model, inputs):
        response = model(inputs).argmax(-1)

        for k in range(len(inputs)):
            self.samples.append(inputs[k])
            self.labels.append(int(response[k]))

    def __call__(self, model: Model, inputs, labels):
        x, y, restore = wrap(inputs, labels)
        del inputs, labels

        adv_samples = []
        for k in range(len(y)):
            while True:
                idx = np.random.randint(len(self.labels))
                if int(y[k].numpy()) != self.labels[idx]:
                    adv_samples.append(ep.astensor(self.samples[idx]).numpy())
                    break

        x = ep.from_numpy(x, np.stack(adv_samples))
        return restore(x)
