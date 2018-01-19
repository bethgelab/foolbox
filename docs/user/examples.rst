========
Examples
========

Here you can find a collection of examples how Foolbox models can be created using different deep learning frameworks and some full-blown attack examples at the end.

Creating a model
================

Keras: ResNet50
---------------

.. code-block:: python3

   import keras
   import numpy as np
   import foolbox

   keras.backend.set_learning_phase(0)
   kmodel = keras.applications.resnet50.ResNet50(weights='imagenet')
   preprocessing = (np.array([104, 116, 123]), 1)
   model = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

   image, _ = foolbox.utils.imagenet_example()
   print(np.argmax(model.predictions(image)))

TensorFlow: VGG19
-----------------

First, create the model in TensorFlow.

.. code-block:: python3

    import tensorflow as tf
    from tensorflow.contrib.slim.nets import vgg
    import numpy as np
    import foolbox

    images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    preprocessed = images - [123.68, 116.78, 103.94]
    logits, _ = vgg.vgg_19(preprocessed, is_training=False)
    restorer = tf.train.Saver(tf.trainable_variables())

    image, _ = foolbox.utils.imagenet_example()

Then transform it into a Foolbox model using one of these four options:

Option 1
^^^^^^^^

This option is recommended if you want to keep the code as short as possible. It makes use
of the TensorFlow session created by Foolbox internally if no default session is set.

.. code-block:: python3

    with foolbox.models.TensorFlowModel(images, logits, (0, 255)) as model:
        restorer.restore(model.session, '/path/to/vgg_19.ckpt')
        print(np.argmax(model.predictions(image)))

Option 2
^^^^^^^^

This option is recommended if you want to create the TensorFlow session yourself.

.. code-block:: python3

    with tf.Session() as session:
        restorer.restore(session, '/path/to/vgg_19.ckpt')
        model = foolbox.models.TensorFlowModel(images, logits, (0, 255))
        print(np.argmax(model.predictions(image)))

Option 3
^^^^^^^^

This option is recommended if you want to avoid nesting context managers, e.g. during interactive development.

.. code-block:: python3

    session = tf.InteractiveSession()
    restorer.restore(session, '/path/to/vgg_19.ckpt')
    model = foolbox.models.TensorFlowModel(images, logits, (0, 255))
    print(np.argmax(model.predictions(image)))
    session.close()

Option 4
^^^^^^^^

This is possible, but usually one of the other options should be preferred.

.. code-block:: python3

    session = tf.Session()
    with session.as_default():
        restorer.restore(session, '/path/to/vgg_19.ckpt')
        model = foolbox.models.TensorFlowModel(images, logits, (0, 255))
        print(np.argmax(model.predictions(image)))
    session.close()

Applying an attack
==================

Once you created a Foolbox model (see the previous section), you can apply an attack.

FGSM (GradientSignAttack)
-------------------------

.. code-block:: python3

   # create a model (see previous section)
   fmodel = ...

   # get source image and label
   image, label = foolbox.utils.imagenet_example()

   # apply attack on source image
   attack  = foolbox.attacks.FGSM(fmodel)
   adversarial = attack(image[:,:,::-1], label)


Creating a targeted adversaial for the Keras ResNet model
=========================================================

.. code-block:: python3

   import foolbox
   from foolbox.models import KerasModel
   from foolbox.attacks import LBFGSAttack
   from foolbox.criteria import TargetClassProbability
   import numpy as np
   import keras
   from keras.applications.resnet50 import ResNet50
   from keras.applications.resnet50 import preprocess_input
   from keras.applications.resnet50 import decode_predictions

   keras.backend.set_learning_phase(0)
   kmodel = ResNet50(weights='imagenet')
   preprocessing = (np.array([104, 116, 123]), 1)
   fmodel = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

   image, label = foolbox.utils.imagenet_example()

   # run the attack
   attack = LBFGSAttack(model=fmodel, criterion=TargetClassProbability(781, p=.5))
   adversarial = attack(image[:, :, ::-1], label)

   # show results
   print(np.argmax(fmodel.predictions(adversarial)))
   print(foolbox.utils.softmax(fmodel.predictions(adversarial))[781])
   adversarial_rgb = adversarial[np.newaxis, :, :, ::-1]
   preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
   print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))

outputs

::

   781
   0.832095
   Top 5 predictions (adversarial:  [[('n04149813', 'scoreboard', 0.83013469), ('n03196217', 'digital_clock', 0.030192226), ('n04152593', 'screen', 0.016133979), ('n04141975', 'scale', 0.011708578), ('n03782006', 'monitor', 0.0091574294)]]
