from typing import Tuple

import numpy as np
import tensorflow as tf
import torch
import math
from eagerpy import astensor_
from .base import T


from eagerpy.tensor import TensorFlowTensor, PyTorchTensor


def rotate_and_shift(inputs: T,
                     restore_type,
                     translation: Tuple[int, int] = (0, 0),
                     rotation: int =0):
    rotation = rotation * math.pi / 180.
    bs = inputs.shape[0]
    theta = np.zeros((2, 3))
    theta[0, :] = [np.cos(rotation), -np.sin(rotation), translation[0]]
    theta[1, :] = [np.sin(rotation), np.cos(rotation), translation[1]]
    theta= np.tile(theta[None], (bs, 1, 1)).reshape(bs, 2, 3)
    if isinstance(inputs, TensorFlowTensor):
        theta = tf.convert_to_tensor(theta)
        tf_tensor = restore_type(inputs)
        transformed_tensor = transform_tf(tf_tensor, theta)
    elif isinstance(inputs, PyTorchTensor):
        torch.tensor(theta)
        pt_tensor = restore_type(inputs)
        transformed_tensor = transform_pt(pt_tensor, theta)
    else:
        raise NotImplementedError()

    return astensor_(transformed_tensor)[0]



def transform_pt(x, theta):
    # x shape: (bs, nch, x, y)
    # angles: scalar or Tensor with (bs,)
    assert len(x.shape) == 4
    assert theta.shape[1:] == (2, 3)

    bs, _, n_x, n_y, = x.shape

    def create_meshgrid(x):
        space_x = torch.linspace(-1, 1, n_x, device=x.device)
        space_y = torch.linspace(-1, 1, n_y, device=x.device)
        meshgrid = torch.meshgrid([space_x, space_y])
        ones = torch.ones(meshgrid[0].shape)
        gridder = torch.stack([meshgrid[1], meshgrid[0], ones], dim=2)
        grid = gridder[None, ...].repeat(bs, 1, 1, 1)[..., None]
        return grid

    meshgrid = create_meshgrid(x)
    theta = theta[:, None, None, :, :].repeat(1, n_x, n_y, 1, 1)
    new_coords = torch.matmul(theta, meshgrid)
    new_coords = new_coords.squeeze_(-1)

    # same as tensorflow addons, for pytorch version >=1.4 align_corners=True
    print("bile")
    transformed_images = torch.nn.functional.grid_sample(
        x, new_coords, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return transformed_images


def shift_and_rotate(images, rotation=0, shifts=(0, 0)):
    pass


# adapted adapted from
# https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
# state @375f990 on 3 Jun 2018
def transform_tf(x, theta):
    """
    Input
    -----
    - x: Images, should be a tensor of shape (bs, n_x, n_y, C).
    - theta: affine transform tensor of shape (bs, 2, 3). Permits cropping,
      translation and isotropic scaling. Initialize to identity matrix.
      It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (bs, n_x, n_y, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    assert theta.shape[1:] == (2, 3)
    assert len(x.shape) == 4
    bs = tf.shape(x)[0]
    n_x = tf.shape(x)[1]  # height matrix
    n_y = tf.shape(x)[2]  # width matrix

    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (bs, n_x, n_y, C)
        - x: flattened tensor of shape (bs*n_x*n_y,)
        - y: flattened tensor of shape (bs*n_x*n_y,)
        Returns
        -------
        - output: tensor of shape (bs, n_x, n_y, C)
        """
        batch_idx = tf.range(0, bs)
        batch_idx = tf.reshape(batch_idx, (bs, 1, 1))
        b = tf.tile(batch_idx, (1, n_x, n_y))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(img, indices)

    def bilinear_sampler(img, x, y):
        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.
        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.
        Input
        -----
        - img: batch of images in (bs, n_x, n_y, C) layout.
        - grid: x, y which is the output of affine_grid_generator.
        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        max_y = tf.cast(n_x - 1, "int32")
        max_x = tf.cast(n_y - 1, "int32")
        zero = tf.zeros([], dtype="int32")

        # rescale x and y to [0, n_y-1/n_x-1]
        x = tf.cast(x, "float32")
        y = tf.cast(y, "float32")
        x = 0.5 * ((x + 1.0) * tf.cast(max_x, "float32"))
        y = 0.5 * ((y + 1.0) * tf.cast(max_y, "float32"))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = tf.cast(tf.floor(x), "int32")
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), "int32")
        y1 = y0 + 1

        # clip to range [0, n_x-1/n_y-1] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = get_pixel_value(img, x0, y0)
        Ib = get_pixel_value(img, x0, y1)
        Ic = get_pixel_value(img, x1, y0)
        Id = get_pixel_value(img, x1, y1)

        # recast as float for delta calculation
        x0 = tf.cast(x0, "float32")
        x1 = tf.cast(x1, "float32")
        y0 = tf.cast(y0, "float32")
        y1 = tf.cast(y1, "float32")

        # calculate deltas
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        return out

    def affine_grid_generator(height, width, theta):
        """
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation [1] of the input feature map.
        Input
        -----
        - height: desired height of grid/output. Used
          to downsample or upsample.
        - width: desired width of grid/output. Used
          to downsample or upsample.
        - theta: affine transform matrices of shape (num_batch, 2, 3).
          For each image in the batch, we have 6 theta parameters of
          the form (2x3) that define the affine transformation T.
        Returns
        -------
        - normalized grid (-1, 1) of shape (num_batch, 2, n_x, n_y).
          The 2nd dimension has 2 components: (x, y) which are the
          sampling points of the original image for each point in the
          target image.
        Note
        ----
        [1]: the affine transformation allows cropping, translation,
             and isotropic scaling.
        """
        num_batch = tf.shape(theta)[0]

        # create normalized 2D grid
        x_l = tf.linspace(-1.0, 1.0, width)
        y_l = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x_l, y_l)

        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = tf.ones_like(x_t_flat)
        sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

        # repeat grid num_batch times
        sampling_grid = tf.expand_dims(sampling_grid, axis=0)
        sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

        # cast to float32 (required for matmul)
        theta = tf.cast(theta, "float32")
        sampling_grid = tf.cast(sampling_grid, "float32")

        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, n_x*n_y)

        # reshape to (num_batch, n_x, n_y, 2)
        batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

        return batch_grids

    batch_grids = affine_grid_generator(n_x, n_y, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    transformed_images = bilinear_sampler(x, x_s, y_s)

    return transformed_images
