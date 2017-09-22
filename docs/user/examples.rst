========
Examples
========

Here you can find a collection of examples how Foolbox models can be created using different deep learning frameworks.

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
