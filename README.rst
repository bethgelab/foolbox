.. image:: https://readthedocs.org/projects/foolbox/badge/?version=latest
    :target: https://foolbox.readthedocs.io/en/latest/

.. image:: https://travis-ci.org/bethgelab/foolbox.svg?branch=master
    :target: https://travis-ci.org/bethgelab/foolbox

.. image:: https://coveralls.io/repos/github/bethgelab/foolbox/badge.svg
    :target: https://coveralls.io/github/bethgelab/foolbox

.. image:: https://badge.fury.io/py/foolbox.svg
    :target: https://badge.fury.io/py/foolbox

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



=======
Foolbox
=======

Foolbox is a Python toolbox to create adversarial examples that fool neural networks. It requires `Python`, `NumPy` and `SciPy`.

Installation
------------

.. code-block:: bash

   # Foolbox 2.0
   pip install foolbox

Foolbox 2.0 requires Python 3.5 or newer.

Documentation
-------------

Documentation for the `latest stable version <https://foolbox.readthedocs.io/en/stable/>`_ as well as
`pre-release versions <https://foolbox.readthedocs.io/en/latest/>`_ is available on ReadTheDocs.

Our paper describing Foolbox is on arXiv: https://arxiv.org/abs/1707.04131

Example
-------

.. code-block:: python

   import foolbox
   import numpy as np
   import torchvision.models as models

   # instantiate model (supports PyTorch, Keras, TensorFlow (Graph and Eager), JAX, MXNet and many more)
   model = models.resnet18(pretrained=True).eval()
   preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
   fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

   # get a batch of images and labels and print the accuracy
   images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
   print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
   # -> 0.9375

   # apply the attack
   attack = foolbox.attacks.FGSM(fmodel)
   adversarials = attack(images, labels)
   # if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
   # if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

   # Foolbox guarantees that all returned adversarials are in fact in adversarials
   print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
   # -> 0.0


.. code-block:: python

   # In rare cases, it can happen that attacks return adversarials that are so close to the decision boundary,
   # that they actually might end up on the other (correct) side if you pass them through the model again like
   # above to get the adversarial class. This is because models are not numerically deterministic (on GPU, some
   # operations such as `sum` are non-deterministic by default) and indepedent between samples (an input might
   # be classified differently depending on the other inputs in the same batch).

   # You can always get the actual adversarial class that was observed for that sample by Foolbox by
   # passing `unpack=False` to get the actual `Adversarial` objects:
   attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.Linf)
   adversarials = attack(images, labels, unpack=False)

   adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
   print(labels)
   print(adversarial_classes)
   print(np.mean(adversarial_classes == labels))  # will always be 0.0

   # The `Adversarial` objects also provide a `distance` attribute. Note that the distances
   # can be 0 (misclassified without perturbation) and inf (attack failed).
   distances = np.asarray([a.distance.value for a in adversarials])
   print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
   print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
   print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))


For more examples, have a look at the `documentation <https://foolbox.readthedocs.io/en/latest/user/examples.html>`__.

Finally, the result can be plotted like this:

.. code-block:: python

   # if you use Jupyter notebooks
   %matplotlib inline

   import matplotlib.pyplot as plt

   image = images[0]
   adversarial = attack(images[:1], labels[:1])[0]

   # CHW to HWC
   image = image.transpose(1, 2, 0)
   adversarial = adversarial.transpose(1, 2, 0)

   plt.figure()

   plt.subplot(1, 3, 1)
   plt.title('Original')
   plt.imshow(image)
   plt.axis('off')

   plt.subplot(1, 3, 2)
   plt.title('Adversarial')
   plt.imshow(adversarial)
   plt.axis('off')

   plt.subplot(1, 3, 3)
   plt.title('Difference')
   difference = adversarial - image
   plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
   plt.axis('off')

   plt.show()


.. image:: https://github.com/bethgelab/foolbox/raw/master/example.png


Interfaces for a range of other deeplearning packages such as TensorFlow 1 and 2,
PyTorch, JAX, Theano, Lasagne and MXNet are available, e.g.

.. code-block:: python

   model = foolbox.models.TensorFlowModel(images, logits, bounds=(0, 255))
   model = foolbox.models.TensorFlowEagerModel(model, bounds=(0, 255))
   model = foolbox.models.PyTorchModel(torchmodel, bounds=(0, 255), num_classes=1000)
   # etc.

Different adversarial criteria such as Top-k, specific target classes or target probability
values for the original class or the target class can be passed to the attack, e.g.

.. code-block:: python

   criterion = foolbox.criteria.TargetClass(22)
   attack    = foolbox.attacks.LBFGSAttack(fmodel, criterion)

Feature requests and bug reports
--------------------------------

We welcome feature requests and bug reports. Just create a new issue on `GitHub <https://github.com/bethgelab/foolbox/issues/new>`__.

Questions & FAQ
---------------

Depending on the nature of your question feel free to post it as an issue on `GitHub <https://github.com/bethgelab/foolbox/issues/new>`__, or post it as a question on `Stack Overflow <https://stackoverflow.com>`_ using the `foolbox` tag. We will try to monitor that tag but if you don't get an answer don't hesitate to contact us.

Before you post a question, please check our `FAQ <https://foolbox.readthedocs.io/en/latest/user/faq.html>`__ and our Documentation on `ReadTheDocs <https://foolbox.readthedocs.io/en/latest/index.html>`__.

Contributions welcome
----------------------

Foolbox is a work in progress and any input is welcome.
Foolbox is particularly well-suited to develop
`new adversarial attacks <https://foolbox.readthedocs.io/en/stable/user/development.html#new-adversarial-attacks>`_
and to support new machine learning and deep learning frameworks by simply adding a wrapper.
By adding reference implementations for adversarial attacks to Foolbox, they will automatically be applicable
to models implemented in any of the supported frameworks such as PyTorch, TensorFlow, Keras, JAX or MxNet.

Citation
--------

If you use Foolbox for your work, please cite our paper:

.. code-block::

  @article{rauber2017foolbox,
    title={Foolbox: A Python toolbox to benchmark the robustness of machine learning models},
    author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
    journal={arXiv preprint arXiv:1707.04131},
    year={2017},
    url={http://arxiv.org/abs/1707.04131},
    archivePrefix={arXiv},
    eprint={1707.04131},
  }

You can find the paper on arXiv: https://arxiv.org/abs/1707.04131

Authors
-------

* `Jonas Rauber <https://github.com/jonasrauber>`_
* `Wieland Brendel <https://github.com/wielandbrendel>`_
