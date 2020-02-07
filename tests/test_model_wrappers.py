import numpy as np

from foolbox import set_seeds
from foolbox.models import ModelWrapper
from foolbox.models import DifferentiableModelWrapper
from foolbox.models import CompositeModel
from foolbox.models import EnsembleAveragedModel


def test_context_manager(gl_bn_model):
    assert isinstance(gl_bn_model, ModelWrapper)
    with gl_bn_model as model:
        assert model is not None
        assert isinstance(model, ModelWrapper)


def test_wrapping(gl_bn_model, bn_image):
    assert isinstance(gl_bn_model, ModelWrapper)
    assert gl_bn_model.num_classes() == 10
    assert np.all(
        gl_bn_model.forward_one(bn_image)
        == gl_bn_model.forward(bn_image[np.newaxis])[0]
    )


def test_diff_wrapper(bn_model, bn_image, bn_label):
    x = bn_image
    la = bn_label
    xs = x[np.newaxis]
    model1 = bn_model
    model2 = DifferentiableModelWrapper(model1)
    assert model1.num_classes() == model2.num_classes()
    assert np.all(model1.forward_one(x) == model2.forward_one(x))
    assert np.all(model1.forward(xs) == model2.forward(xs))
    assert np.all(model1.gradient_one(x, la) == model2.gradient_one(x, la))
    assert np.all(
        model1.forward_and_gradient_one(x, la)[0]
        == model2.forward_and_gradient_one(x, la)[0]
    )
    assert np.all(
        model1.forward_and_gradient_one(x, la)[1]
        == model2.forward_and_gradient_one(x, la)[1]
    )
    assert np.all(
        model1.forward_and_gradient(np.array([x]), [la])[0]
        == model2.forward_and_gradient(np.array([x]), [la])[0]
    )
    assert np.all(
        model1.forward_and_gradient(np.array([x]), [la])[1]
        == model2.forward_and_gradient(np.array([x]), [la])[1]
    )
    g = model1.forward_one(x)
    assert np.all(model1.backward_one(g, x) == model2.backward_one(g, x))


def test_composite_model(gl_bn_model, bn_model, bn_image, bn_label):
    num_classes = 10
    test_grad = np.random.rand(num_classes).astype(np.float32)
    model = CompositeModel(gl_bn_model, bn_model)
    with model:
        assert gl_bn_model.num_classes() == model.num_classes()
        assert np.all(gl_bn_model.forward_one(bn_image) == model.forward_one(bn_image))
        assert np.all(
            bn_model.gradient_one(bn_image, bn_label)
            == model.gradient_one(bn_image, bn_label)
        )
        assert np.all(
            bn_model.backward_one(test_grad, bn_image)
            == model.backward_one(test_grad, bn_image)
        )
        assert np.all(
            gl_bn_model.forward_one(bn_image)
            == model.forward_and_gradient_one(bn_image, bn_label)[0]
        )
        assert np.all(
            bn_model.forward_and_gradient_one(bn_image, bn_label)[1]
            == model.forward_and_gradient_one(bn_image, bn_label)[1]
        )
        assert np.all(
            bn_model.forward_and_gradient(np.array([bn_image]), [bn_label])[0]
            == model.forward_and_gradient(np.array([bn_image]), [bn_label])[0]
        )
        assert np.all(
            bn_model.forward_and_gradient(np.array([bn_image]), [bn_label])[1]
            == model.forward_and_gradient(np.array([bn_image]), [bn_label])[1]
        )


def test_estimate_gradient_wrapper(eg_bn_adversarial, bn_image):
    p, ia = eg_bn_adversarial.forward_one(bn_image)
    set_seeds(22)
    g = eg_bn_adversarial.gradient_one(bn_image)
    set_seeds(22)
    p2, g2, ia2 = eg_bn_adversarial.forward_and_gradient_one(bn_image)
    set_seeds(22)
    p3, g3, ia3 = eg_bn_adversarial.forward_and_gradient(np.array([bn_image]))
    p3, g3, ia3 = p3[0], g3[0], ia3[0]

    assert np.all(p == p2) and np.all(p == p3)
    assert np.all(g == g2) and np.all(g == g3)
    assert ia == ia2 == ia3


def test_ensemble_average_wrapper(
    sn_bn_model, bn_model, avg_sn_bn_adversarial, bn_image, bn_label
):
    p, ia = avg_sn_bn_adversarial.forward_one(bn_image)
    set_seeds(22)
    g = avg_sn_bn_adversarial.gradient_one(bn_image)
    set_seeds(22)
    p2, g2, ia2 = avg_sn_bn_adversarial.forward_and_gradient_one(bn_image)
    set_seeds(22)
    p3, g3, ia3 = avg_sn_bn_adversarial.forward_and_gradient(np.array([bn_image]))
    p3, g3, ia3 = p3[0], g3[0], ia3[0]

    assert np.all(p == p2) and np.all(p == p3)
    assert np.all(g == g2) and np.all(g == g3)
    assert ia == ia2 == ia3

    num_classes = 10
    test_grad = np.random.rand(num_classes).astype(np.float32)
    model = EnsembleAveragedModel(sn_bn_model, ensemble_size=200)
    with model:
        assert sn_bn_model.num_classes() == model.num_classes()

        assert np.allclose(
            bn_model.forward_one(bn_image), model.forward_one(bn_image), atol=1e-3
        )

        assert np.allclose(
            bn_model.gradient_one(bn_image, bn_label),
            model.gradient_one(bn_image, bn_label),
            atol=1e-3,
        )

        assert np.allclose(
            bn_model.backward_one(test_grad, bn_image),
            model.backward_one(test_grad, bn_image),
            atol=1e-3,
        )
        assert np.allclose(
            bn_model.forward_one(bn_image),
            model.forward_and_gradient_one(bn_image, bn_label)[0],
            atol=1e-3,
        )
        assert np.allclose(
            bn_model.forward_and_gradient_one(bn_image, bn_label)[1],
            model.forward_and_gradient_one(bn_image, bn_label)[1],
            atol=1e-3,
        )
        assert np.allclose(
            bn_model.forward_and_gradient(np.array([bn_image]), [bn_label])[0],
            model.forward_and_gradient(np.array([bn_image]), [bn_label])[0],
            atol=1e-3,
        )
        assert np.allclose(
            bn_model.forward_and_gradient(np.array([bn_image]), [bn_label])[1],
            model.forward_and_gradient(np.array([bn_image]), [bn_label])[1],
            atol=1e-3,
        )


def test_batched_estimate_gradient_wrapper(eg_bn_model, bn_image, bn_label):
    set_seeds(22)
    g1 = eg_bn_model.gradient(bn_image[np.newaxis], [bn_label])[0]
    set_seeds(22)
    g2 = eg_bn_model.gradient_one(bn_image, bn_label)

    assert np.allclose(g1, g2)

    gs = eg_bn_model.gradient(np.array([bn_image] * 2), np.array([bn_label] * 2))
    assert np.allclose(gs[0], gs[1], atol=1e-1)
