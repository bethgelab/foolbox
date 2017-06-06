============
Development
============

New Adversarial Attacks
=======================

Simply subclass the :class:`Attack` class and implement the :meth:`_apply` method. Use the :class:`Adversarial` instance passed as the first argument to calculate predictions, to check if an image is adversarial, and to get gradients if necessary. The :class:`Adversarial` class keeps track of the best adversarial automatically, there is no need to determine it manually in the implementation of the attack.
