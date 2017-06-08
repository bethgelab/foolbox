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
   FGSM
   LBFGSAttack
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


.. rubric:: :doc:`attacks/approxgradient`

.. autosummary::
   :nosignatures:

   ApproximateLBFGSAttack


.. rubric:: :doc:`attacks/other`

.. autosummary::
   :nosignatures:

   PrecomputedImagesAttack
