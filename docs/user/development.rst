============
Development
============

To install Foolbox in editable mode, see the installation instructions under :ref:`dev-install`.

Running Tests
=======================

pytest
``````

To run the tests, you need to have `pytest <https://docs.pytest.org/en/latest/getting-started.html>`_ and `pytest-cov <http://pytest-cov.readthedocs.io/en/latest/readme.html#installation>`_ installed. Afterwards, you can simply run ``pytest`` in the root folder of the project. Some tests will require TensorFlow, PyTorch and the other frameworks, so to run all tests, you need to have all of them installed.

flake8
``````
Foolbox follows the `PEP 8 style guide for Python code <https://www.python.org/dev/peps/pep-0008/>`_. To check for violations, we use `flake8 <http://flake8.pycqa.org/en/latest/>`_ and run it like this:

.. code-block:: sh

   flake8 --ignore E402,E741 .

New Adversarial Attacks
=======================

Foolbox makes it easy to develop new adversarial attacks that can be applied to arbitrary models.

To implement an attack, simply subclass the :class:`Attack` class, implement the :meth:`__call__` method and decorate it with the :decorator:`call_decorator`. The :decorator:`call_decorator` will make sure that your :meth:`__call__` implementation will be called with an instance of the :class:`Adversarial` class. You can use this instance to ask for model predictions and gradients, get the original image and its label and more. In addition, the :class:`Adversarial` instance automatically keeps track of the best adversarial amongst all the images tested by the attack. That way, the implementation of the attack can focus on the attack logic.
