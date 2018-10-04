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
   NewtonFoolAttack
   DeepFoolL2Attack
   DeepFoolLinfinityAttack
   ADefAttack
   SLSQPAttack
   SaliencyMapAttack
   IterativeGradientAttack
   IterativeGradientSignAttack
   CarliniWagnerAttack


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
