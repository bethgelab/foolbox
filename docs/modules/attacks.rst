:mod:`foolbox.attacks`
================================

.. automodule:: foolbox.attacks

.. toctree::
   :hidden:

   attacks/gradient
   attacks/blackbox
   attacks/approxgradient
   attacks/other


.. rubric:: :doc:`attacks/gradient`

.. autosummary::
   :nosignatures:

   GradientSignAttack
   IterativeGradientSignAttack
   GradientAttack
   IterativeGradientAttack
   FGSM
   LBFGSAttack
   DeepFoolAttack
   DeepFool


.. rubric:: :doc:`attacks/blackbox`

.. autosummary::
   :nosignatures:

   SaliencyMapAttack
   GaussianBlurAttack
   ContrastReductionAttack
   SinglePixelAttack
   LocalSearchAttack
   SLSQPAttack
   AdditiveUniformNoiseAttack
   AdditiveGaussianNoiseAttack
   SaltAndPepperNoiseAttack


.. rubric:: :doc:`attacks/approxgradient`

.. autosummary::
   :nosignatures:

   ApproximateLBFGSAttack


.. rubric:: :doc:`attacks/other`

.. autosummary::
   :nosignatures:

   PrecomputedImagesAttack
