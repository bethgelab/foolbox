.. raw:: html

   <a href="https://foolbox.jonasrauber.de"><img src="https://raw.githubusercontent.com/bethgelab/foolbox/master/guide/.vuepress/public/logo_small.png" align="right" /></a>

.. image:: https://badge.fury.io/py/foolbox.svg
   :target: https://badge.fury.io/py/foolbox

.. image:: https://readthedocs.org/projects/foolbox/badge/?version=latest
    :target: https://foolbox.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

.. image:: https://joss.theoj.org/papers/10.21105/joss.02607/status.svg
   :target: https://doi.org/10.21105/joss.02607

===============================================================================================================================
Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX
===============================================================================================================================

`Foolbox <https://foolbox.jonasrauber.de>`_ is a **Python library** that lets you easily run adversarial attacks against machine learning models like deep neural networks. It is built on top of EagerPy and works natively with models in `PyTorch <https://pytorch.org>`_, `TensorFlow <https://www.tensorflow.org>`_, and `JAX <https://github.com/google/jax>`_.

🔥 Design 
----------

**Foolbox 3** a.k.a. **Foolbox Native** has been rewritten from scratch
using `EagerPy <https://github.com/jonasrauber/eagerpy>`_ instead of
NumPy to achieve native performance on models
developed in PyTorch, TensorFlow and JAX, all with one code base without code duplication.

- **Native Performance**: Foolbox 3 is built on top of EagerPy and runs natively in PyTorch, TensorFlow, and JAX and comes with real batch support.
- **State-of-the-art attacks**: Foolbox provides a large collection of state-of-the-art gradient-based and decision-based adversarial attacks.
- **Type Checking**: Catch bugs before running your code thanks to extensive type annotations in Foolbox.

📖 Documentation
-----------------

- **Guide**: The best place to get started with Foolbox is the official `guide <https://foolbox.jonasrauber.de>`_.
- **Tutorial**: If you are looking for a tutorial, check out this `Jupyter notebook <https://github.com/jonasrauber/foolbox-native-tutorial/blob/master/foolbox-native-tutorial.ipynb>`_ |colab|.
- **Documentation**: The API documentation can be found on `ReadTheDocs <https://foolbox.readthedocs.io/en/stable/>`_.

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/jonasrauber/foolbox-native-tutorial/blob/master/foolbox-native-tutorial.ipynb

🚀 Quickstart
--------------

.. code-block:: bash

   pip install foolbox

Foolbox is tested with Python 3.9 and newer - however, it will most likely also work with version 3.6 - 3.8. To use it with `PyTorch <https://pytorch.org>`_, `TensorFlow <https://www.tensorflow.org>`_, or `JAX <https://github.com/google/jax>`_, the respective framework needs to be installed separately. These frameworks are not declared as dependencies because not everyone wants to use and thus install all of them and because some of these packages have different builds for different architectures and CUDA versions. Besides that, all essential dependencies are automatically installed.

You can see the versions we currently use for testing in the `Compatibility section <#-compatibility>`_ below, but newer versions are in general expected to work.

🎉 Example
-----------

.. code-block:: python

   import foolbox as fb

   model = ...
   fmodel = fb.PyTorchModel(model, bounds=(0, 1))

   attack = fb.attacks.LinfPGD()
   epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
   _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)


More examples can be found in the `examples <./examples/>`_ folder, e.g.
a full `ResNet-18 example <./examples/single_attack_pytorch_resnet18.py>`_.

📄 Citation
------------

If you use Foolbox for your work, please cite our `JOSS paper on Foolbox Native <https://doi.org/10.21105/joss.02607>`_ and our `ICML workshop paper on Foolbox <https://arxiv.org/abs/1707.04131>`_ using the following BibTeX entries:

.. code-block::

   @article{rauber2017foolboxnative,
     doi = {10.21105/joss.02607},
     url = {https://doi.org/10.21105/joss.02607},
     year = {2020},
     publisher = {The Open Journal},
     volume = {5},
     number = {53},
     pages = {2607},
     author = {Jonas Rauber and Roland Zimmermann and Matthias Bethge and Wieland Brendel},
     title = {Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, and JAX},
     journal = {Journal of Open Source Software}
   }

.. code-block::

   @inproceedings{rauber2017foolbox,
     title={Foolbox: A Python toolbox to benchmark the robustness of machine learning models},
     author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
     booktitle={Reliable Machine Learning in the Wild Workshop, 34th International Conference on Machine Learning},
     year={2017},
     url={http://arxiv.org/abs/1707.04131},
   }


👍 Contributions
-----------------

We welcome contributions of all kind, please have a look at our
`development guidelines <https://foolbox.jonasrauber.de/guide/development.html>`_.
In particular, you are invited to contribute
`new adversarial attacks <https://foolbox.jonasrauber.de/guide/adding_attacks.html>`_.
If you would like to help, you can also have a look at the issues that are
marked with `contributions welcome
<https://github.com/bethgelab/foolbox/issues?q=is%3Aopen+is%3Aissue+label%3A%22contributions+welcome%22>`_.

💡 Questions?
--------------

If you have a question or need help, feel free to open an issue on GitHub.
Once GitHub Discussions becomes publically available, we will switch to that.

💨 Performance
--------------

Foolbox Native is much faster than Foolbox 1 and 2. A basic `performance comparison`_ can be found in the `performance` folder.

🐍 Compatibility
-----------------

We currently test with the following versions:

* PyTorch 1.4.0
* TensorFlow 2.1.0
* JAX 0.1.57
* NumPy 1.18.1

.. _performance comparison: performance/README.md
