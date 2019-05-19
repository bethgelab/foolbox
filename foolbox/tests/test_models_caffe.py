import pytest
import numpy as np

from foolbox.models import CaffeModel


@pytest.mark.parametrize("bn_model_caffe, num_classes",
                         [(10, 10), (1000, 1000)],
                         indirect=["bn_model_caffe"])
def test_caffe_model(bn_model_caffe, num_classes):
    model = bn_model_caffe
    test_images = np.random.rand(2, num_classes, 5, 5).astype(np.float32)
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


def test_caffe_model_gradient(tmpdir):
    import caffe
    from caffe import layers as L

    bounds = (0, 255)
    channels = num_classes = 1000

    net_spec = caffe.NetSpec()
    net_spec.data = L.Input(name="data",
                            shape=dict(dim=[1, num_classes, 5, 5]))
    net_spec.reduce_1 = L.Reduction(net_spec.data,
                                    reduction_param={"operation": 4,
                                                     "axis": 3})
    net_spec.output = L.Reduction(net_spec.reduce_1,
                                  reduction_param={"operation": 4,
                                                   "axis": 2})
    net_spec.label = L.Input(name="label",
                             shape=dict(dim=[1]))
    net_spec.loss = L.SoftmaxWithLoss(net_spec.output, net_spec.label)
    wf = tmpdir.mkdir("test_models_caffe")\
               .join("test_caffe_model_gradient_proto_{}.prototxt"
                     .format(num_classes))
    wf.write("force_backward: true\n" + str(net_spec.to_proto()))
    preprocessing = (np.arange(num_classes)[:, None, None],
                     np.random.uniform(size=(channels, 5, 5)) + 1)
    net = caffe.Net(str(wf), caffe.TEST)
    model = CaffeModel(
        net,
        bounds=bounds,
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


@pytest.mark.parametrize("bn_model_caffe, num_classes",
                         [(10, 10), (1000, 1000)],
                         indirect=["bn_model_caffe"])
def test_caffe_backward(bn_model_caffe, num_classes):
    model = bn_model_caffe
    test_image = np.random.rand(num_classes, 5, 5).astype(np.float32)
    test_grad_pre = np.random.rand(num_classes).astype(np.float32)

    test_grad = model.backward_one(test_grad_pre, test_image)
    assert test_grad.shape == test_image.shape

    manual_grad = np.repeat(np.repeat(
        (test_grad_pre / 25.).reshape((-1, 1, 1)),
        5, axis=1), 5, axis=2)

    np.testing.assert_almost_equal(
        test_grad,
        manual_grad)


def test_caffe_model_preprocessing_shape_change(tmpdir):
    import caffe
    from caffe import layers as L

    bounds = (0, 255)
    channels = num_classes = 1000

    net_spec = caffe.NetSpec()
    net_spec.data = L.Input(name="data",
                            shape=dict(dim=[1, num_classes, 5, 5]))
    net_spec.reduce_1 = L.Reduction(net_spec.data,
                                    reduction_param={"operation": 4,
                                                     "axis": 3})
    net_spec.output = L.Reduction(net_spec.reduce_1,
                                  reduction_param={"operation": 4, "axis": 2})
    net_spec.label = L.Input(name="label", shape=dict(dim=[1]))
    net_spec.loss = L.SoftmaxWithLoss(net_spec.output, net_spec.label)
    wf = tmpdir.mkdir("test_models_caffe")\
               .join("test_caffe_model_preprocessing_shape_change_{}.prototxt"
                     .format(num_classes))
    wf.write("force_backward: true\n" + str(net_spec.to_proto()))
    net = caffe.Net(str(wf), caffe.TEST)
    model1 = CaffeModel(
        net,
        bounds=bounds)

    def preprocessing2(x):
        if x.ndim == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        elif x.ndim == 4:
            x = np.transpose(x, axes=(0, 3, 1, 2))

        def grad(dmdp):
            assert dmdp.ndim == 3
            dmdx = np.transpose(dmdp, axes=(1, 2, 0))
            return dmdx

        return x, grad

    model2 = CaffeModel(
        net,
        bounds=bounds,
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
