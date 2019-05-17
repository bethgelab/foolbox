=========
Model Zoo
=========

This tutorial will show you how the model zoo can be used to run your attack against a robust model.

Downloading a model
===================

For this tutorial, we will download the `Analysis by Synthesis` model implemented in `PyTorch`
and run a `FGSM (GradienSignAttack)` against it.

.. code-block:: python3

    from foolbox import zoo

    # download the model
    model = zoo.get_model(url="https://github.com/bethgelab/AnalysisBySynthesis")

    # read image and label
    image = ...
    label = ...

    # apply attack on source image
    attack  = foolbox.attacks.FGSM(model)
    adversarial = attack(image, label)
