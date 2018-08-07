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

   GradientAttack
   GradientSignAttack
   FGSM
   LinfinityBasicIterativeAttack
   BasicIterativeMethod
   BIM
   L1BasicIterativeAttack
   L2BasicIterativeAttack
   ProjectedGradientDescentAttack
   ProjectedGradientDescent
   RandomStartProjectedGradientDescentAttack
   RandomProjectedGradientDescent
   RandomPGD
   MomentumIterativeAttack
   MomentumIterativeMethod
   LBFGSAttack
   DeepFoolAttack
   DeepFoolL2Attack
   DeepFoolLinfinityAttack
   SLSQPAttack
   SaliencyMapAttack
   IterativeGradientAttack
   IterativeGradientSignAttack


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
   SaltAndPepperNoiseAttack
   BlendedUniformNoiseAttack
   PointwiseAttack


.. rubric:: :doc:`attacks/other`

.. autosummary::
   :nosignatures:

   BinarizationRefinementAttack
   PrecomputedImagesAttack
