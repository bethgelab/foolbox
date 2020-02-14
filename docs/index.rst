Welcome to Foolbox Native
=========================

Foolbox is a Python toolbox to create adversarial examples that fool neural networks.
*Foolbox 3.0* a.k.a. *Foolbox Native* has been completely rewritten from scratch.
It is now built on top of `EagerPy <https://github.com/jonasrauber/eagerpy>`_
and comes with native support for these frameworks:

* `PyTorch <https://pytorch.org>`_
* `TensorFlow <https://www.tensorflow.org>`_
* `JAX <https://github.com/google/jax>`_

Foolbox comes with a :doc:`large collection of adversarial attacks <modules/attacks>`, both gradient-based white-box attacks as well as decision-based and score-based black-box attacks.

The source code and a `minimal working example <https://github.com/bethgelab/foolbox#example>`_ can be found on `GitHub <https://github.com/bethgelab/foolbox>`_.


.. toctree::
   :maxdepth: 2
   :caption: User API

   modules/models
   modules/attacks
   modules/criteria
   modules/distances
   modules/utils
   modules/plot
   modules/zoo

.. toctree::
   :maxdepth: 2
   :caption: Internal API

   modules/devutils
   modules/tensorboard
   modules/types


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
