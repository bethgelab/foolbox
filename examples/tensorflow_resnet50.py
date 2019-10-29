import foolbox.ext.native as fbn
import tensorflow as tf


if __name__ == "__main__":
    # instantiate a model
    model = tf.keras.applications.ResNet50(weights="imagenet")
    pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel = fbn.models.TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)

    # get data and test the model
    images, labels = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    print(fbn.utils.accuracy(fmodel, images, labels))

    # apply the attack
    attack = fbn.attacks.LinfinityBasicIterativeAttack(fmodel)
    adversarials = attack(images, labels, epsilon=0.03 * 255.0, step_size=0.005 * 255.0)
    print(fbn.utils.accuracy(fmodel, adversarials, labels))

    # apply another attack
    attack = fbn.attacks.L2BasicIterativeAttack(fmodel)
    adversarials = attack(images, labels, epsilon=2.0 * 255.0, step_size=0.2 * 255.0)
    print(fbn.utils.accuracy(fmodel, adversarials, labels))
