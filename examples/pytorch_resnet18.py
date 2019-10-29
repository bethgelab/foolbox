import foolbox.ext.native as fbn
import torchvision.models as models


if __name__ == "__main__":
    # instantiate a model
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fbn.models.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get data and test the model
    images, labels = fbn.utils.samples(fmodel, dataset="imagenet", batchsize=16)
    print(fbn.utils.accuracy(fmodel, images, labels))

    # apply the attack
    attack = fbn.attacks.LinfinityBasicIterativeAttack(fmodel)
    adversarials = attack(images, labels, epsilon=0.03, step_size=0.005)  # L-inf norm
    print(fbn.utils.accuracy(fmodel, adversarials, labels))

    # apply another attack
    attack = fbn.attacks.L2BasicIterativeAttack(fmodel)
    adversarials = attack(images, labels, epsilon=2.0, step_size=0.2)  # L2 norm
    print(fbn.utils.accuracy(fmodel, adversarials, labels))
