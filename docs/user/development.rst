============
Development
============

To install Foolbox in editable mode, see the installation instructions under :ref:`dev-install`.

.. _development:

Running Tests
=======================

pytest
``````

To run the tests, you need to have `pytest <https://docs.pytest.org/en/latest/getting-started.html>`_ and `pytest-cov <http://pytest-cov.readthedocs.io/en/latest/readme.html#installation>`_ installed. Afterwards, you can simply run ``pytest`` in the root folder of the project. Some tests will require TensorFlow, PyTorch and the other frameworks, so to run all tests, you need to have all of them installed. Note however that this can take quite long (Foolbox has many tests) and installing all frameworks with the correct versions is difficult due to conflicting dependencies. You can also open a pull-request and then we will run all the tests using travis.

Style Guide
===========

We use `Black <https://black.readthedocs.io/>`_ to format all code in a consistent and PEP-8 conform way.
All pull-requests are checked using both ``black`` and ``flake8``. Simply install ``black`` and run ``black .`` after
all your changes or ideally even on each commit using `pre-commit <https://black.readthedocs.io/en/stable/version_control_integration.html>`_.

New Adversarial Attacks
=======================

Foolbox makes it easy to develop new adversarial attacks that can be applied to arbitrary models.

To implement an attack, simply subclass the :class:`Attack` class, implement the :meth:`__call__` method and decorate it with the :decorator:`call_decorator`. The :decorator:`call_decorator` will make sure that your :meth:`__call__` implementation will be called with an instance of the :class:`Adversarial` class. You can use this instance to ask for model predictions and gradients, get the original image and its label and more. In addition, the :class:`Adversarial` instance automatically keeps track of the best adversarial amongst all the inputs tested by the attack. That way, the implementation of the attack can focus on the attack logic.

To implement an attack that can make use of the batch support introduced in Foolbox 2.0, implement the :meth:`as_generator` method and decorate it with the :decorator:`generator_decorator`. All model calls using the :class:`Adversarial` object should use ``yield``.
