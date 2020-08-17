# Getting Started

## Installation

You can install the latest release from [PyPI](https://pypi.org/project/foolbox/) using `pip`:

```bash
python3 -m pip install foolbox
```

Foolbox requires Python 3.6 or newer. To use it with [PyTorch](https://pytorch.org), [TensorFlow](https://www.tensorflow.org), or [JAX](https://github.com/google/jax), the respective framework needs to be installed separately. These frameworks are not declared as dependencies because not everyone wants to use and thus install all of them and because some of these packages have different builds for different architectures and CUDA versions. Besides that, all essential dependencies are automatically installed.

::: warning NOTE
Foolbox requires Python 3.6 or newer.
:::

## Getting a Model

Once Foolbox is installed, you need to turn your PyTorch, TensorFlow, or JAX model into a Foolbox model.

### PyTorch

For PyTorch, you simply instantiate your `torch.nn.Module` and then pass it
to `fb.PyTorchModel`. Here we use a pretrained ResNet-18 from `torchvision`.
Additionally, you should specify the preprocessing expected by the model
(e.g. subtracting `mean`, and dividing by `std`, along the third axis from the back)
and the bounds of the input space (before the preprocessing).

```python
# PyTorch ResNet18
import torch
import torchvision
model = torchvision.models.resnet18(pretrained=True)
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
```

### TensorFlow

For TensorFlow, you simply instantiate your `tf.keras.Model` and then pass it
to `fb.TensorFlowModel`. Here we show three examples using pretrained ImageNet
models. Additionally, you should specify the preprocessing expected by the model
(e.g. flipping an axis, here from RGB to BGR, subtracting `mean`,
and dividing by `std`, along the third axis from the back)
and the bounds of the input space (before the preprocessing).

```python
# TensorFlow ResNet50
import tensorflow as tf
model = tf.keras.applications.ResNet50(weights="imagenet")
preprocessing = dict(flip_axis=-1, mean=[103.939, 116.779, 123.68])
bounds = (0, 255)
fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
```

```python
# TensorFlow ResNet50V2
import tensorflow as tf
model = tf.keras.applications.ResNet50V2(weights="imagenet")
preprocessing = dict()
bounds = (-1, 1)
fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
```

```
# TensorFlow MobileNetV2
import tensorflow as tf
model = tf.keras.applications.MobileNetV2(weights="imagenet")
preprocessing = dict()
bounds = (-1, 1)
fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
```

### JAX

For JAX, you simply specify your model as a callable, i.e. an instance of a
class with a `__call__` method or a simple function. It should take an
input array and return the array with predictions. You can then pass this
callable to `fb.JAXModel`. Additionally, you should specify the
preprocessing (see previous examples) and
the bounds of the input space (before the preprocessing).

```python
class Model:
    def __call__(self, x):
        # turn the inputs x into predictions y
        y = x  # replace with your real model
        return y

model = Model()
bounds = (0, 1)
fmodel = fbn.JAXModel(model, bounds)
```

## Transform Bounds

Next you can optionally transform the bounds of the input space of our model.
In the following, we want to work with a model that has (0, 1) bounds.

```python
fmodel = fmodel.transform_bounds((0, 1))
```

If your model already had bounds `(0, 1)`, this does not do anything.
But if your model had different bounds, e.g. `(0, 255)` this would adjust
the preprocessing accordingly such that your model now expects inputs
between `0` and `1`. This is particularly useful if you work with
different models that have different bounds.

## Dataset

Before we can attack our model, we first need some data.
For convenience, Foolbox comes with helper functions that provide
a small set of sample images from different computer vision datasets.

```python
images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
```

Note that images and labels should be a batch of native tensors, i.e.
PyTorch tensors, TensorFlow tensors, or JAX arrays, depending on which framework
you use.

## Attacking the Model

Now we have everything ready to attack the model. Before we do that,
we will quickly check its clean accuracy on our evaluation set.

```python
fb.utils.accuracy(fmodel, images, labels)
# -> 0.9375 (depends on the model!)
```

To run an attack, we first instantiate the corresponding class.

```python
attack = fb.attacks.LinfDeepFoolAttack()
```

And finally we can apply the attack on our model by passing
the input tensor (here `images`), the corresponding true `labels`,
and one or more `epsilons`.

```python
raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
```

The attack returns three tensors.

1. The raw adversarial examples. This depends on the attack and we cannot make an guarantees about this output.
2. The clipped adversarial examples. These are guaranteed to not be perturbed more than epsilon and thus are the actual adversarial examples you want to visualize. Note that some of them might not actually switch the class. To know which samples are actually adversarial, you should look at the third tensor.
3. The third tensor contains a boolean for each sample, indicating which samples are true adversarials that are both misclassified and within the epsilon balls around the clean samples.

How to use these tensors will become more clear in a moment.

## Multiple Epsilons

Usually, you should not just look at a single epsilon, but at many different epislons from small to large.
The most efficient way to obtain the corresponding results is by
running the attack with multiple epsilons. It will automatically
select the right strategy depending on the type of attack.

```python
import numpy as np
epsilons = np.linspace(0.0, 0.005, num=20)
```

Let's rerun the attack for all `epsilons`.

```python
raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
```

The returned tensors, `raw`, `clipped`, and `is_adv` now have an additional batch dimension for the different `epsilons`.

## Robust Accuracy

You can now obtain the robust accuracy by simply averaging `is_adv`.

```python
robust_accuracy = 1 - is_adv.float32().mean(axis=-1)
```

You can now plot the robust accuracy using Matplotlib.

```python
import matplotlib.pyplot as plt
plt.plot(epsilons, robust_accuracy.numpy())
```

You can also visualize the adversarials using `fb.plot.images`.

## Learn More

To learn more, have a look at our [Tutorial](https://github.com/jonasrauber/foolbox-native-tutorial/blob/master/foolbox-native-tutorial.ipynb),
the [examples](./examples.md), the [API docs](https://foolbox.readthedocs.io/en/stable/) and of course the [README](https://github.com/bethgelab/foolbox).

