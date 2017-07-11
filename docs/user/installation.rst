============
Installation
============

Foolbox is a Python package to create adversarial examples. We test using Python 2.7, 3.5 and 3.6, but other versions of Python might work as well. **We recommend using Python 3!**.

Stable release
==============

You can install the latest stable release of Foolbox from PyPI using `pip`:

.. code-block:: bash

   pip install foolbox

Make sure that `pip` installs packages for Python 3, otherwise you might need to use `pip3` instead of `pip`.

Development version
===================

Alternatively, you can install the latest development version of Foolbox from GitHub. We try to keep the master branch stable, so this version should usually work fine. Feel free to open an issue on GitHub if you encounter any problems.

.. code-block:: bash

   pip install https://github.com/bethgelab/foolbox/archive/master.zip


.. _dev-install:

Contributing to Foolbox
=======================

If you would like to contribute the development of Foolbox, install it in editable mode:

.. code-block:: bash

   git clone https://github.com/bethgelab/foolbox.git
   cd foolbox
   pip install --editable .

To contribute your changes, you will need to fork the Foolbox repository on GitHub.
You can than add it as a remote:

.. code-block:: bash

   git remote rename origin upstream
   git remote add origin https://github.com/<your-github-name>/foolbox.git

You can now commit your changes, push them to your fork and create a pull-request to
contribute them to Foolbox.
