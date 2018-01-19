:mod:`foolbox.attacks`
================================

.. automodule:: foolbox.attacks

.. toctree::
   :hidden:

   attacks/gradient
   attacks/score
   attacks/decision
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


.. rubric:: :doc:`attacks/score`

.. autosummary::
   :nosignatures:

   SinglePixelAttack
   LocalSearchAttack
   ApproximateLBFGSAttack


.. rubric:: :doc:`attacks/decision`

.. autosummary::
   :nosignatures:

   BoundaryAttack
   GaussianBlurAttack
   ContrastReductionAttack
   AdditiveUniformNoiseAttack
   AdditiveGaussianNoiseAttack
   BlendedUniformNoiseAttack
   SaltAndPepperNoiseAttack
   ResetAttack


.. rubric:: :doc:`attacks/other`

.. autosummary::
   :nosignatures:

   PrecomputedImagesAttack
