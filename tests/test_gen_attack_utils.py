import eagerpy as ep

from foolbox.attacks.gen_attack_utils import rescale_images


def test_pytorch_numpy_compatibility() -> None:
    import numpy as np
    import torch

    x_np = np.random.uniform(0.0, 1.0, size=(16, 3, 64, 64))
    x_torch = torch.from_numpy(x_np)

    x_np_ep = ep.astensor(x_np)
    x_torch_ep = ep.astensor(x_torch)

    x_up_np_ep = rescale_images(x_np_ep, (16, 3, 128, 128), 1)
    x_up_torch_ep = rescale_images(x_torch_ep, (16, 3, 128, 128), 1)

    x_up_np = x_up_np_ep.numpy()
    x_up_torch = x_up_torch_ep.numpy()

    assert np.allclose(x_up_np, x_up_torch)


def test_pytorch_numpy_compatibility_different_axis() -> None:
    import numpy as np
    import torch

    x_np = np.random.uniform(0.0, 1.0, size=(16, 64, 64, 3))
    x_torch = torch.from_numpy(x_np)

    x_np_ep = ep.astensor(x_np)
    x_torch_ep = ep.astensor(x_torch)

    x_up_np_ep = rescale_images(x_np_ep, (16, 128, 128, 3), -1)
    x_up_torch_ep = rescale_images(x_torch_ep, (16, 128, 128, 3), -1)

    x_up_np = x_up_np_ep.numpy()
    x_up_torch = x_up_torch_ep.numpy()

    assert np.allclose(x_up_np, x_up_torch)


def test_pytorch_tensorflow_compatibility() -> None:
    import numpy as np
    import torch
    import tensorflow as tf

    x_np = np.random.uniform(0.0, 1.0, size=(16, 3, 64, 64))
    x_torch = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np)

    x_tf_ep = ep.astensor(x_tf)
    x_torch_ep = ep.astensor(x_torch)

    x_up_tf_ep = rescale_images(x_tf_ep, (16, 3, 128, 128), 1)
    x_up_torch_ep = rescale_images(x_torch_ep, (16, 3, 128, 128), 1)

    x_up_tf = x_up_tf_ep.numpy()
    x_up_torch = x_up_torch_ep.numpy()

    assert np.allclose(x_up_tf, x_up_torch)


def test_jax_numpy_compatibility() -> None:
    import numpy as np
    import jax.numpy as jnp

    x_np = np.random.uniform(0.0, 1.0, size=(16, 3, 64, 64))
    x_jax = jnp.array(x_np)

    x_np_ep = ep.astensor(x_np)
    x_jax_ep = ep.astensor(x_jax)

    x_up_np_ep = rescale_images(x_np_ep, (16, 3, 128, 128), 1)
    x_up_jax_ep = rescale_images(x_jax_ep, (16, 3, 128, 128), 1)

    x_up_np = x_up_np_ep.numpy()
    x_up_jax = x_up_jax_ep.numpy()

    assert np.allclose(x_up_np, x_up_jax)
