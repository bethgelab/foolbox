:mod:`foolbox.v1.attacks`
================================

.. automodule:: foolbox.v1.attacks

.. toctree::
   :hidden:

   v1_attacks/gradient
   v1_attacks/score
   v1_attacks/decision
   v1_attacks/other


.. rubric:: :doc:`v1_attacks/gradient`

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
   PGD
   RandomStartProjectedGradientDescentAttack
   RandomProjectedGradientDescent
   RandomPGD
   AdamL1BasicIterativeAttack
   AdamL2BasicIterativeAttack
   AdamProjectedGradientDescentAttack
   AdamProjectedGradientDescent
   AdamPGD
   AdamRandomStartProjectedGradientDescentAttack
   AdamRandomProjectedGradientDescent
   AdamRandomPGD
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
   CarliniWagnerL2Attack
   EADAttack
   DecoupledDirectionNormL2Attack
   SparseFoolAttack

.. rubric:: :doc:`v1_attacks/score`

.. autosummary::
   :nosignatures:

   SinglePixelAttack
   LocalSearchAttack
   ApproximateLBFGSAttack


.. rubric:: :doc:`v1_attacks/decision`

.. autosummary::
   :nosignatures:

   BoundaryAttack
   SpatialAttack
   PointwiseAttack
   GaussianBlurAttack
   ContrastReductionAttack
   AdditiveUniformNoiseAttack
   AdditiveGaussianNoiseAttack
   SaltAndPepperNoiseAttack
   BlendedUniformNoiseAttack
   BoundaryAttackPlusPlus
   HopSkipJumpAttack


.. rubric:: :doc:`v1_attacks/other`

.. autosummary::
   :nosignatures:

   BinarizationRefinementAttack
   PrecomputedAdversarialsAttack
