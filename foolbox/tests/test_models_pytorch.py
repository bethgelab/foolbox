import pytest
import numpy as np
import torch

from foolbox.models import PyTorchModel


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_pytorch_model(num_classes):
    import torch
    import torch.nn as nn

    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
    test_label = 7

    assert model.forward(test_images).shape \
        == (2, num_classes)

    test_logits = model.forward_one(test_images[0])
    assert test_logits.shape == (num_classes,)

    test_gradient = model.gradient_one(test_images[0], test_label)
    assert test_gradient.shape == test_images[0].shape

    np.testing.assert_almost_equal(
        model.forward_and_gradient_one(test_images[0], test_label)[0],
        test_logits)
    np.testing.assert_almost_equal(
        model.forward_and_gradient_one(test_images[0], test_label)[1],
        test_gradient)

    assert model.num_classes() == num_classes


def test_pytorch_model_preprocessing():
    import torch
    import torch.nn as nn

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model1 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    model2 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        preprocessing=preprocessing)

    model3 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    np.random.seed(22)
    test_images = np.random.rand(2, channels, 5, 5).astype(np.float32)
    test_images_copy = test_images.copy()

    p1 = model1.forward(test_images)
    p2 = model2.forward(test_images)

    # make sure the images have not been changed by
    # the in-place preprocessing
    assert np.all(test_images == test_images_copy)

    p3 = model3.forward(test_images)

    assert p1.shape == p2.shape == p3.shape == (2, num_classes)

    np.testing.assert_array_almost_equal(
        p1 - p1.max(),
        p3 - p3.max(),
        decimal=5)


def test_pytorch_model_gradient():
    import torch
    import torch.nn as nn

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)

    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        preprocessing=preprocessing)

    epsilon = 1e-2

    np.random.seed(23)
    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_label = 7

    _, g1 = model.forward_and_gradient_one(test_image, test_label)

    l1 = model._loss_fn(test_image - epsilon / 2 * g1, test_label)
    l2 = model._loss_fn(test_image + epsilon / 2 * g1, test_label)

    assert 1e4 * (l2 - l1) > 1

    # make sure that gradient is numerically correct
    np.testing.assert_array_almost_equal(
        1e4 * (l2 - l1),
        1e4 * epsilon * np.linalg.norm(g1)**2,
        decimal=1)


@pytest.mark.parametrize('num_classes', [10, 1000])
def test_pytorch_backward(num_classes):
    import torch
    import torch.nn as nn

    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()
    model = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    test_image = np.random.rand(channels, 5, 5).astype(np.float32)
    test_grad_pre = np.random.rand(num_classes).astype(np.float32)

    test_grad = model.backward_one(test_grad_pre, test_image)
    assert test_grad.shape == test_image.shape

    manual_grad = np.repeat(np.repeat(
        (test_grad_pre / 25.).reshape((-1, 1, 1)),
        5, axis=1), 5, axis=2)

    np.testing.assert_almost_equal(
        test_grad,
        manual_grad)


def test_pytorch_model_preprocessing_shape_change():
    import torch
    import torch.nn as nn

    num_classes = 1000
    bounds = (0, 255)
    channels = num_classes

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            logits = x
            return logits

    model = Net()

    model1 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes)

    def preprocessing2(x):
        if x.ndim == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        elif x.ndim == 4:
            x = np.transpose(x, axes=(0, 3, 1, 2))

        def grad(dmdp):
            if dmdp.ndim == 3:
                dmdx = np.transpose(dmdp, axes=(1, 2, 0))
            elif dmdp.ndim == 4:
                dmdx = np.transpose(dmdp, axes=(0, 2, 3, 1))
            else:
                raise AssertionError
            return dmdx

        return x, grad

    model2 = PyTorchModel(
        model,
        bounds=bounds,
        num_classes=num_classes,
        preprocessing=preprocessing2)

    np.random.seed(22)
    test_images_nhwc = np.random.rand(2, 5, 5, channels).astype(np.float32)
    test_images_nchw = np.transpose(test_images_nhwc, (0, 3, 1, 2))

    p1 = model1.forward(test_images_nchw)
    p2 = model2.forward(test_images_nhwc)

    assert np.all(p1 == p2)

    p1 = model1.forward_one(test_images_nchw[0])
    p2 = model2.forward_one(test_images_nhwc[0])

    assert np.all(p1 == p2)

    g1 = model1.gradient_one(test_images_nchw[0], 3)
    assert g1.ndim == 3
    g1 = np.transpose(g1, (1, 2, 0))
    g2 = model2.gradient_one(test_images_nhwc[0], 3)

    np.testing.assert_array_almost_equal(g1, g2)


def test_pytorch_device(bn_model_pytorch):
    m = bn_model_pytorch
    model1 = PyTorchModel(
        m._model,
        bounds=m.bounds(),
        num_classes=m.num_classes(),
        device='cpu')
    model2 = PyTorchModel(
        m._model,
        bounds=m.bounds(),
        num_classes=m.num_classes(),
        device=torch.device('cpu'))
    assert model1.device == model2.device
