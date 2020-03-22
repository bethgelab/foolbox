---
home: true
heroImage: /logo.png
heroText: Foolbox
tagline: A Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, and JAX
actionText: Get Started →
actionLink: /guide/
features:
- title: Native Performance
  details: Foolbox 3 is built on top of EagerPy and runs natively in PyTorch, TensorFlow, JAX, and NumPy.
- title: State-of-the-art attacks
  details: Foolbox provides a large collection of state-of-the-art gradient-based and decision-based adversarial attacks.
- title: Type Checking
  details: Catch bugs before running your code thanks to extensive type annotations in Foolbox.
footer: Copyright © 2020 Jonas Rauber

---

### What is Foolbox?

**Foolbox** is a **Python library** that let's you easily run adversarial attacks against machine learning models like deep neural networks. It is built on top of [**EagerPy**](https://eagerpy.jonasrauber.de) and works natively with models in [**PyTorch**](https://pytorch.org), [**TensorFlow**](https://www.tensorflow.org), [**JAX**](https://github.com/google/jax), and [**NumPy**](https://numpy.org).

```python
import foolbox as fb

model = ...
fmodel = fb.PyTorchModel(model)

attack = fb.attacks.LinfPGD()
epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)
```
