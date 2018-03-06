============
Development
============

To install Foolbox in editable mode, see the installation instructions under :ref:`dev-install`.

Running Tests
=======================

To run the tests, you need to have `pytest <https://docs.pytest.org/en/latest/getting-started.html>`_ installed. Afterwards, you can simply run ``pytest`` in the root folder of the project.

Foolbox follows the `PEP 8 style guide for Python code <https://www.python.org/dev/peps/pep-0008/>`. To check for violations, we use `flake8 <http://flake8.pycqa.org/en/latest/>`_ and run it like this:

.. code-block:: sh

   flake8 --ignore E402,E741 .

New Adversarial Attacks
=======================

Foolbox makes it easy to develop new adversarial attacks that can be applied to arbitrary models.

To implement an attack, simply subclass the :class:`Attack` class and implement the :meth:`_apply` method. The :meth:`_apply` method will be called with an instance of the :class:`Adversarial` class. You can use this instance to ask for model predictions and gradients, get the original image and its label and more. In addition, the :class:`Adversarial` instance automatically keeps track of the best adversarial amongst all the images tested by the attack. That way, the implementation of the attack can focus on the attack logic.
