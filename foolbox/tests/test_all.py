import numpy as np
import tensorflow as tf

import foolbox


def test_tensorflow(image):
    # vgg with random weights

    g = tf.Graph()
    with g.as_default():
        images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        net = tf.contrib.layers.flatten(images)
        net = tf.layers.dense(net, 1000)
        net = net / 100
        logits = net
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        sess.run(init_op)
        model = foolbox.models.TensorFlowModel(images, logits, bounds=(0, 255))
        criterion = foolbox.criteria.Misclassification()
        attack = foolbox.attacks.FGSM(model, criterion)

        label = np.argmax(model.predictions(image))
        adversarial = attack(image=image, label=label, unpack=False)

        assert adversarial.best_distance().value() < 5
        adv_label = np.argmax(model.predictions(adversarial.get()))
        assert label != adv_label

        # the same using a manually created Adversarial instance
        adversarial = foolbox.Adversarial(model, criterion, image, label)
        attack = foolbox.attacks.FGSM()
        attack(adversarial)

        assert adversarial.best_distance().value() < 5
        adv_label = np.argmax(model.predictions(adversarial.get()))
        assert label != adv_label
