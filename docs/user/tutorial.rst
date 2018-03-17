========
Tutorial
========

This tutorial will show you how an adversarial attack can be used to find adversarial examples for a model.

Creating a model
================

For the tutorial, we will target `VGG19` implemented in `TensorFlow`, but it is straight forward to apply the same to other models or other frameworks such as `Theano` or `PyTorch`.

.. code-block:: python3

   import tensorflow as tf

   images = tf.placeholder(tf.float32, (None, 224, 224, 3))
   preprocessed = vgg_preprocessing(images)
   logits = vgg19(preprocessed)

To turn a model represented as a standard TensorFlow graph into a model that can be attacked by the Adversarial Toolbox, all we have to do is to create a new `TensorFlowModel` instance:

.. code-block:: python3

   from foolbox.models import TensorFlowModel

   model = TensorFlowModel(images, logits, bounds=(0, 255))


Specifying the criterion
========================

To run an adversarial attack, we need to specify the type of adversarial we are looking for. This can be done using the :class:`Criterion` class.

.. code-block:: python3

   from foolbox.criteria import TargetClassProbability

   target_class = 22
   criterion = TargetClassProbability(target_class, p=0.99)

Running the attack
==================

Finally, we can create and apply the attack:


.. code-block:: python3

   from foolbox.attacks import LBFGSAttack

   attack = LBFGSAttack(model, criterion)

   image = np.asarray(Image.open('example.jpg'))
   label = np.argmax(model.predictions(image))

   adversarial = attack(image, label=label)


Visualizing the adversarial examples
====================================

To plot the adversarial example we can use `matplotlib`:

.. code-block:: python3

  import matplotlib.pyplot as plt

  plt.subplot(1, 3, 1)
  plt.imshow(image)

  plt.subplot(1, 3, 2)
  plt.imshow(adversarial)

  plt.subplot(1, 3, 3)
  plt.imshow(adversarial - image)
