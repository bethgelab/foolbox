========
Examples
========

Here you can find a collection of examples how Foolbox models can be created using different deep learning frameworks and some full-blown attack examples at the end.

Running an attack
=================

Running a batch attack against a PyTorch model
----------------------------------------------

.. code-block:: python3

   import foolbox
   import numpy as np
   import torchvision.models as models

   # instantiate model (supports PyTorch, Keras, TensorFlow (Graph and Eager), MXNet and many more)
   model = models.resnet18(pretrained=True).eval()
   preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
   fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

   # get a batch of images and labels and print the accuracy
   images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
   print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
   # -> 0.9375

   # apply the attack
   attack = foolbox.attacks.FGSM(fmodel)
   adversarials = attack(images, labels)
   # if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
   # if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

   # Foolbox guarantees that all returned adversarials are in fact in adversarials
   print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
   # -> 0.0

   # ---

   # In rare cases, it can happen that attacks return adversarials that are so close to the decision boundary,
   # that they actually might end up on the other (correct) side if you pass them through the model again like
   # above to get the adversarial class. This is because models are not numerically deterministic (on GPU, some
   # operations such as `sum` are non-deterministic by default) and indepedent between samples (an input might
   # be classified differently depending on the other inputs in the same batch).

   # You can always get the actual adversarial class that was observed for that sample by Foolbox by
   # passing `unpack=False` to get the actual `Adversarial` objects:
   attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.Linf)
   adversarials = attack(images, labels, unpack=False)

   adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
   print(labels)
   print(adversarial_classes)
   print(np.mean(adversarial_classes == labels))  # will always be 0.0

   # The `Adversarial` objects also provide a `distance` attribute. Note that the distances
   # can be 0 (misclassified without perturbation) and inf (attack failed).
   distances = np.asarray([a.distance.value for a in adversarials])
   print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
   print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
   print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))


Running an attack on single sample against a Keras model
--------------------------------------------------------

.. code-block:: python3

   import foolbox
   import keras
   import numpy as np
   from keras.applications.resnet50 import ResNet50

   # instantiate model
   keras.backend.set_learning_phase(0)
   kmodel = ResNet50(weights='imagenet')
   preprocessing = dict(flip_axis=-1, mean=np.array([104, 116, 123]))  # RGB to BGR and mean subtraction
   fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

   # get source image and label
   image, label = foolbox.utils.imagenet_example()

   # apply attack on source image
   attack = foolbox.v1.attacks.FGSM(fmodel)
   adversarial = attack(image, label)
   # if the attack fails, adversarial will be None and a warning will be printed


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
   preprocessing = dict(flip_axis=-1, mean=np.array([104, 116, 123]))  # RGB to BGR and mean subtraction
   model = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

   image, label = foolbox.utils.imagenet_example()
   print(np.argmax(model.forward_one(image)), label)

PyTorch: ResNet18
-----------------

You might be interested in checking out the full PyTorch example at the end
of this document.

.. code-block:: python3

   import torchvision.models as models
   import numpy as np
   import foolbox

   # instantiate the model
   resnet18 = models.resnet18(pretrained=True).cuda().eval()  # for CPU, remove cuda()
   mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
   std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
   model = foolbox.models.PyTorchModel(resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

   image, label = foolbox.utils.imagenet_example(data_format='channels_first')
   image = image / 255
   print(np.argmax(model.forward_one(image)), label)

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
        print(np.argmax(model.forward_one(image)))

Option 2
^^^^^^^^

This option is recommended if you want to create the TensorFlow session yourself.

.. code-block:: python3

    with tf.Session() as session:
        restorer.restore(session, '/path/to/vgg_19.ckpt')
        model = foolbox.models.TensorFlowModel(images, logits, (0, 255))
        print(np.argmax(model.forward_one(image)))

Option 3
^^^^^^^^

This option is recommended if you want to avoid nesting context managers, e.g. during interactive development.

.. code-block:: python3

    session = tf.InteractiveSession()
    restorer.restore(session, '/path/to/vgg_19.ckpt')
    model = foolbox.models.TensorFlowModel(images, logits, (0, 255))
    print(np.argmax(model.forward_one(image)))
    session.close()

Option 4
^^^^^^^^

This is possible, but usually one of the other options should be preferred.

.. code-block:: python3

    session = tf.Session()
    with session.as_default():
        restorer.restore(session, '/path/to/vgg_19.ckpt')
        model = foolbox.models.TensorFlowModel(images, logits, (0, 255))
        print(np.argmax(model.forward_one(image)))
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
   attack  = foolbox.v1.attacks.FGSM(fmodel)
   adversarial = attack(image, label)


Creating an untargeted adversarial for a PyTorch model
======================================================

.. code-block:: python3

   import foolbox
   import torch
   import torchvision.models as models
   import numpy as np

   # instantiate the model
   resnet18 = models.resnet18(pretrained=True).eval()
   if torch.cuda.is_available():
       resnet18 = resnet18.cuda()
   mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
   std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
   fmodel = foolbox.models.PyTorchModel(
       resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

   # get source image and label
   image, label = foolbox.utils.imagenet_example(data_format='channels_first')
   image = image / 255.  # because our model expects values in [0, 1]

   print('label', label)
   print('predicted class', np.argmax(fmodel.forward_one(image)))

   # apply attack on source image
   attack = foolbox.v1.attacks.FGSM(fmodel)
   adversarial = attack(image, label)

   print('adversarial class', np.argmax(fmodel.forward_one(adversarial)))

outputs

::

   label 282
   predicted class 282
   adversarial class 281

To plot image and adversarial, don't forget to move the channel
axis to the end before passing them to matplotlib's imshow, e.g.
using ``np.transpose(image, (1, 2, 0))``.


Creating a targeted adversarial for the Keras ResNet model
==========================================================

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
   preprocessing = dict(flip_axis=-1, mean=np.array([104, 116, 123]))  # RGB to BGR and mean subtraction
   fmodel = KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

   image, label = foolbox.utils.imagenet_example()

   # run the attack
   attack = LBFGSAttack(model=fmodel, criterion=TargetClassProbability(781, p=.5))
   adversarial = attack(image, label)

   # show results
   print(np.argmax(fmodel.forward_one(adversarial)))
   print(foolbox.utils.softmax(fmodel.forward_one(adversarial))[781])
   preds = kmodel.predict(preprocess_input(adversarial[np.newaxis].copy()))
   print("Top 5 predictions (adversarial: ", decode_forward_one(preds, top=5))

outputs

::

   781
   0.832095
   Top 5 predictions (adversarial:  [[('n04149813', 'scoreboard', 0.83013469), ('n03196217', 'digital_clock', 0.030192226), ('n04152593', 'screen', 0.016133979), ('n04141975', 'scale', 0.011708578), ('n03782006', 'monitor', 0.0091574294)]]
