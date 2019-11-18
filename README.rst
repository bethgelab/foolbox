.. image:: https://badge.fury.io/py/foolbox-native.svg
    :target: https://badge.fury.io/py/foolbox-native

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


==============
Foolbox Native
==============

Foolbox Native is an extension for `Foolbox <https://github.com/bethgelab/foolbox>`_
that tries to bring native performance to Foolbox. This extension is a
prototype with the goal of ultimately becoming part of Foolbox itself.
Please be aware of the the differences to Foolbox listed below.

Foolbox Native currently provides full support for:

* PyTorch
* TensorFlow
* JAX

Other frameworks can be used as well by falling back to Foolbox for the
model API, see below.

Installation
------------

.. code-block:: bash

   pip install --upgrade foolbox-native


PyTorch Example
---------------

.. code-block:: python

   import foolbox.ext.native as fbn
   import torchvision.models as models

   # instantiate a model
   model = models.resnet18(pretrained=True).eval()
   preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
   fmodel = fbn.models.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

   # get data and test the model
   images, labels = fbn.utils.samples(fmodel, dataset='imagenet', batchsize=16)
   print(fbn.utils.accuracy(fmodel, images, labels))
   # -> 0.9375

   # apply the attack
   attack = fbn.attacks.LinfinityBasicIterativeAttack(fmodel)
   adversarials = attack(images, labels, epsilon=0.03, step_size=0.005)  # L-inf norm
   print(fbn.utils.accuracy(fmodel, adversarials, labels))
   # -> 0.0

   # apply another attack
   attack = fbn.attacks.L2BasicIterativeAttack(fmodel)
   adversarials = attack(images, labels, epsilon=2.0, step_size=0.2)  # L2 norm
   print(fbn.utils.accuracy(fmodel, adversarials, labels))
   # -> 0.0

TensorFlow Example
------------------

.. code-block:: python

   import foolbox.ext.native as fbn
   import tensorflow as tf

   # instantiate a model
   model = tf.keras.applications.ResNet50(weights='imagenet')
   preprocessing = dict(flip_axis=-1, mean=[104., 116., 123.])  # RGB to BGR
   fmodel = fbn.models.TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)

   # get data and test the model
   images, labels = fbn.utils.samples(fmodel, dataset='imagenet', batchsize=16)
   print(fbn.utils.accuracy(fmodel, images, labels))

   # apply the attack
   attack = fbn.attacks.LinfinityBasicIterativeAttack(fmodel)
   adversarials = attack(images, labels, epsilon=0.03 * 255., step_size=0.005 * 255.)  # L-inf norm
   print(fbn.utils.accuracy(fmodel, adversarials, labels))

   # apply another attack
   attack = fbn.attacks.L2BasicIterativeAttack(fmodel)
   adversarials = attack(images, labels, epsilon=2.0 * 255., step_size=0.2 * 255.)  # L2 norm
   print(fbn.utils.accuracy(fmodel, adversarials, labels))

Robust Accuracy Evaluation
--------------------------

.. code-block:: python

   import foolbox.ext.native as fbn

   # get fmodel, images, labels like above
   fmodel = ...
   images, labels = ...

   attacks = [
       L2BasicIterativeAttack,
       L2CarliniWagnerAttack,
       L2ContrastReductionAttack,
       BinarySearchContrastReductionAttack,
       LinearSearchContrastReductionAttack,
   ]
   epsilons = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

   _, robust_accuracy = fbn.evaluate_l2(fmodel, x, y, attacks=attacks, epsilons=epsilons)
   print(robust_accuracy)

   # Plot an accuracy-distortion curve
   plt.plot(epsilons, robust_accuracy)

Other Frameworks
----------------

Foolbox Native supports all frameworks supported by the standard Foolbox
by simply wrapping the ``foolbox.models.*`` classes using ``fbn.model.FoolboxModel``.
This, however, comes with a performance penalty. Nevertheless, it still
allows one to profit from the manually batched attack reimplementations
that come with Foolbox Native.

Important differences to Foolbox
--------------------------------

Unlike Foolbox:

* Foolbox Native does not yet guarantee API stability (expect breaking changes)

* Foolbox Native is currently limited to very few attacks

* Foolbox Native does not make any guarantees about the output of an attack

  * The user is responsible for checking if the returned samples are adversarial

  * Whether the size of the perturbations is guaranteed depends on the attack

  * Foolbox, on the other hand, searches for the smallest perturbation while
    guaranteeing that the returned samples are adversarial
