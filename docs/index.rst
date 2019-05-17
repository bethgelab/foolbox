Welcome to Foolbox
==================

Foolbox is a Python toolbox to create adversarial examples that fool neural networks.

It comes with support for many frameworks to build models including

* TensorFlow
* PyTorch
* Theano
* Keras
* Lasagne
* MXNet

and it is easy to extend to other frameworks.

In addition, it comes with a **large collection of adversarial attacks**, both gradient-based attacks as well as black-box attacks. See :doc:`modules/attacks` for details.

The source code and a `minimal working example`_ can be found on `GitHub`_.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/installation
   user/tutorial
   user/examples
   user/adversarial
   user/zoo
   user/development
   user/faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/models
   modules/criteria
   modules/zoo
   modules/distances
   modules/attacks
   modules/adversarial
   modules/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/bethgelab/foolbox
.. _minimal working example: https://github.com/bethgelab/foolbox#example
