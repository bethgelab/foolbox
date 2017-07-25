========
Examples
========

Here you can find a collection of examples how Foolbox models can be created using different deep learning frameworks.

Creating a model
================

Keras: ResNet50
---------------

.. code-block:: python3
   import keras
   import numpy as np
   import foolbox

   keras.backend.set_learning_phase(0)
   kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
   preprocessing = (np.array([104, 116, 123]), 1)
   fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

Applying an attack
==================

Once you created a Foolbox model (see the previous section), you can apply an attack.

FGSM (GradientSignAttack)
-------------------------

.. code-block:: python3
   # create a model (see previous section)
   fmodel = ...

   # get source image and label
   image, label = foolbox.utils.imagenet_example()

   # apply attack on source image
   attack  = foolbox.attacks.FGSM(fmodel)
   adversarial = attack(image[:,:,::-1], label)
