========
Advanced
========

The :class:`Adversarial` class provides an advanced way to specify the adversarial example that should be found by an attack and provides detailed information about the created adversarial. In addition, it provides a way to improve a previously found adversarial example by re-running an attack.


Implicit
========

.. code-block:: python3

  model = TensorFlowModel(images, logits, bounds=(0, 255))
  criterion = TargetClassProbability('ostrich', p=0.99)
  attack = LBFGSAttack(model, criterion)

Running the attack by passing image and label will
implicitly create an :class:`Adversarial` instance. By
passing `unpack=False` we tell the attack to return the
:class:`Adversarial` instance rather than the actual image.

.. code-block:: python3

   adversarial = attack(image, label=label, unpack=False)

We can then get the actual image using the :attr:`image` attribute:

.. code-block:: python3

   adversarial_image = adversarial.image


Explicit
========

.. code-block:: python3

   model = TensorFlowModel(images, logits, bounds=(0, 255))
   criterion = TargetClassProbability('ostrich', p=0.99)
   attack = LBFGSAttack()

We can also create the :class:`Adversarial` instance ourselves
and then pass it to the attack.

.. code-block:: python3

   adversarial = Adversarial(model, criterion, image, label)
   attack(adversarial)

Again, we can get the image using the :attr:`image` attribute:

.. code-block:: python3

   adversarial_image = adversarial.image

This approach gives us more flexibility and allows us to specify
a different distance measure:

.. code-block:: python3

   distance = MeanAbsoluteDistance
   adversarial = Adversarial(model, criterion, image, label, distance=distance)
