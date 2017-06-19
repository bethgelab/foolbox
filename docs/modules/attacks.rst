:mod:`foolbox.attacks`
================================

.. automodule:: foolbox.attacks

.. toctree::
   :hidden:

   attacks/gradient
   attacks/blackbox
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
   SLSQPAttack
   SaliencyMapAttack


.. rubric:: :doc:`attacks/blackbox`

.. autosummary::
   :nosignatures:

   GaussianBlurAttack
   ContrastReductionAttack
   SinglePixelAttack
   LocalSearchAttack
   AdditiveUniformNoiseAttack
   AdditiveGaussianNoiseAttack
   SaltAndPepperNoiseAttack
   ApproximateLBFGSAttack


.. rubric:: :doc:`attacks/other`

.. autosummary::
   :nosignatures:

   PrecomputedImagesAttack
