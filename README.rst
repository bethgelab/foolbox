.. image:: https://readthedocs.org/projects/foolbox/badge/?version=latest
    :target: https://foolbox.readthedocs.io/en/latest/

.. image:: https://travis-ci.org/bethgelab/foolbox.svg?branch=master
    :target: https://travis-ci.org/bethgelab/foolbox

.. image:: https://coveralls.io/repos/github/bethgelab/foolbox/badge.svg
    :target: https://coveralls.io/github/bethgelab/foolbox



=======
Foolbox
=======

Foolbox is a Python toolbox to create adversarial examples that fool neural networks. It requires `Python`, `NumPy` and `SciPy`.

Installation
------------

.. code-block:: bash

   pip install foolbox

We test using Python 2.7, 3.5 and 3.6. Other Python versions might work as well. **We recommend using Python 3!**

Documentation
-------------

Documentation is available on readthedocs: http://foolbox.readthedocs.io/

Our paper describing Foolbox is on arXiv: https://arxiv.org/abs/1707.04131

Example
-------

.. code-block:: python

   import foolbox
   import keras
   import numpy
   from keras.applications.resnet50 import ResNet50

   # instantiate model
   keras.backend.set_learning_phase(0)
   kmodel = ResNet50(weights='imagenet')
   preprocessing = (numpy.array([104, 116, 123]), 1)
   fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

   # get source image and label
   image, label = foolbox.utils.imagenet_example()

   # apply attack on source image
   attack  = foolbox.attacks.FGSM(fmodel)
   adversarial = attack(image[:,:,::-1], label)

Interfaces for a range of other deeplearning packages such as TensorFlow, 
PyTorch, Theano, Lasagne and MXNet are available, e.g.

.. code-block:: python

   model = foolbox.models.TensorFlowModel(images, logits, bounds=(0, 255))
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

Questions
---------

Depending on the nature of your question feel free to post it as an issue on `GitHub <https://github.com/bethgelab/foolbox/issues/new>`__, or post it as a question on `Stack Overflow <https://stackoverflow.com>`_ using the `foolbox` tag. We will try to monitor that tag but if you don't get an answer don't hesitate to contact us.

Contributions welcome
----------------------

Foolbox is a work in progress and any input is welcome.

In particular, we encourage users of deep learning frameworks for which we do not yet have builtin support, e.g. Caffe, Caffe2 or CNTK, to contribute the necessary wrappers. Don't hestiate to contact us if we can be of any help.

Moreoever, attack developers are encouraged to share their reference implementation using Foolbox so that it will be available to everyone.

Citation
--------

If you find Foolbox useful for your scientific work, please consider citing it
in resulting publications:

.. code-block::

  @article{rauber2017foolbox
    title={Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine learning models},
    author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
    journal={arXiv preprint},
    eprint={1707.04131},
    year={2017}
  }

You can find the paper on arXiv: https://arxiv.org/abs/1707.04131

Authors
-------

* `Jonas Rauber <https://github.com/jonasrauber>`_
* `Wieland Brendel <https://github.com/wielandbrendel>`_

