.. image:: https://readthedocs.org/projects/foolbox/badge/?version=latest
    :target: https://foolbox.readthedocs.io/en/latest/

.. image:: https://travis-ci.org/bethgelab/foolbox.svg?branch=master
    :target: https://travis-ci.org/bethgelab/foolbox

.. image:: https://coveralls.io/repos/github/bethgelab/foolbox/badge.svg
    :target: https://coveralls.io/github/bethgelab/foolbox



=======
Foolbox
=======

Foolbox is a Python toolbox to create adversarial examples that fool neural networks. It requires `Python 3`, `NumPy` and `SciPy`.

Installation
------------

.. code-block:: bash

   pip install foolbox

Documentation
-------------

Documentation is available on readthedocs: http://foolbox.readthedocs.io/

Example
-------

.. code-block:: python

   import foolbox
   import keras
   from keras.applications.resnet50 import ResNet50, preprocess_input

   # instantiate model
   keras.backend.set_learning_phase(0)
   kmodel = ResNet50(weights='imagenet')
   fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocess_fn=preprocess_input)

   # get source image and label
   image, label = foolbox.utils.imagenet_example()

   # apply attack on source image
   attack  = foolbox.attacks.FGSM(fmodel)
   adv_img = attack(image=image, label=label)

Interfaces for a range of other deeplearning packages such as TensorFlow, 
PyTorch and Lasagne are available, e.g.

.. code-block:: python

   model = foolbox.models.PyTorchModel(torchmodel)

Different adversarial criteria such as Top-k, specific target classes or target probability 
levels can be passed to the attack, e.g.

.. code-block:: python

   criterion = foolbox.criteria.TargetClass(22)
   attack    = foolbox.attacks.FGSM(fmodel, criterion)

Development
-----------

Foolbox is a work in progress and any input is welcome.

Authors
-------

* Jonas Rauber
* Wieland Brendel
