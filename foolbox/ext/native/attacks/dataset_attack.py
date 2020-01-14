import eagerpy as ep
import numpy as np


class DatasetAttack:
    """This is a helper attack that makes it straight-forward to choose initialisation points
       for boundary-type attacks from a given data set. All it does is to store samples and
       the predicted responses of a given model in order to select suitable adversarial images
       from the given data set.
    """

    def __init__(self, model):
        self.model = model
        self.samples = []
        self.labels = []

    def feed(self, inputs):
        response = self.model.forward(inputs).argmax(1)

        for k in range(len(inputs)):
            self.labels.append(int(response[k]))
            self.samples.append(inputs[k])

    def __call__(self, inputs, labels):
        inputs = ep.astensor(inputs)
        labels = ep.astensor(labels)
        x = ep.zeros_like(inputs)

        adv_samples = []

        for k in range(len(labels)):
            while True:
                idx = np.random.randint(len(self.labels))
                if int(labels[k].numpy()) != self.labels[idx]:
                    adv_samples.append(ep.astensor(self.samples[idx]).numpy())
                    break

        x = ep.from_numpy(x, np.stack(adv_samples))
        return x.tensor
