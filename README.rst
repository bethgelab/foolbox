.. raw:: html

   <a href="https://foolbox.jonasrauber.de"><img src="https://raw.githubusercontent.com/bethgelab/foolbox/master/guide/.vuepress/public/logo_small.png" align="right" /></a>

.. image:: https://badge.fury.io/py/foolbox.svg
   :target: https://badge.fury.io/py/foolbox

.. image:: https://readthedocs.org/projects/foolbox/badge/?version=latest
    :target: https://foolbox.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

=========================================================================================================================
Foolbox Native: A Python toolbox to create adversarial examples that fool neural networks in PyTorch, TensorFlow, and JAX
=========================================================================================================================

`Foolbox <https://foolbox.jonasrauber.de>`_ is a **Python library** that let's you easily run adversarial attacks against machine learning models like deep neural networks. It is built on top of EagerPy and works natively with models in `PyTorch <https://pytorch.org>`_, `TensorFlow <https://www.tensorflow.org>`_, `JAX <https://github.com/google/jax>`_, and `NumPy <https://numpy.org>`_.

🔥 Design 
----------

**Foolbox 3** a.k.a. **Foolbox Native** has been rewritten from scratch
using `EagerPy <https://github.com/jonasrauber/eagerpy>`_ instead of
NumPy to achieve native performance on models
developed in PyTorch, TensorFlow and JAX, all with one code base.

- **Native Performance**: Foolbox 3 is built on top of EagerPy and runs natively in PyTorch, TensorFlow, JAX, and NumPyand comes with real batch support.
- **State-of-the-art attacks**: Foolbox provides a large collection of state-of-the-art gradient-based and decision-based adversarial attacks.
- **Type Checking**: Catch bugs before running your code thanks to extensive type annotations in Foolbox.

📖 Documentation
-----------------

- **Guide**: The best place to get started with Foolbox is the official `guide <https://foolbox.jonasrauber.de>`_.
- **Tutorial**: If you are looking for a tutorial, check out this `Jupyter notebook <https://github.com/jonasrauber/foolbox-native-tutorial/blob/master/foolbox-native-tutorial.ipynb>`_.
- **Documentaiton**: Finally, you can find the full API documentation on `ReadTheDocs <https://foolbox.readthedocs.io/en/stable/>`_.

🚀 Quickstart
--------------

.. code-block:: bash

   pip install foolbox


🎉 Example
-----------

.. code-block:: python

   import foolbox as fb

   model = ...
   fmodel = fb.PyTorchModel(model)

   attack = fb.attacks.LinfPGD()
   epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
   _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)


More examples can be found in the `examples <./examples/>`_ folder, e.g.
a full `ResNet-18 example <./examples/pytorch_resnet18.py>`_.

📄 Citation
------------

If you use Foolbox for your work, please cite our `paper <https://arxiv.org/abs/1707.04131>`_ using the this BibTex entry:

.. code-block::

   @inproceedings{rauber2017foolbox,
     title={Foolbox: A Python toolbox to benchmark the robustness of machine learning models},
     author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
     booktitle={Reliable Machine Learning in the Wild Workshop, 34th International Conference on Machine Learning},
     year={2017},
     url={http://arxiv.org/abs/1707.04131},
   }


🐍 Compatibility
-----------------

We currently test with the following versions:

* PyTorch 1.4.0
* TensorFlow 2.1.0
* JAX 0.1.57
* NumPy 1.18.1
