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

Robust Vision Benchmark
==================

.. image:: http://bethgelab.org/media/banners/benchmark_banner.png
    :target: https://robust.vision/benchmark

You might want to have a look at our recently announced `Robust Vision Benchmark <https://robust.vision/benchmark>`__, a benchmark for adversarial attacks and the robustness of machine learning models.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/installation
   user/tutorial
   user/examples
   user/adversarial
   user/development

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules/models
   modules/criteria
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
