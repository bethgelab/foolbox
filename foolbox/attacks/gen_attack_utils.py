from typing import Union, List, Tuple
import eagerpy as ep


def rescale_jax(x: ep.JAXTensor, target_shape: List[int]) -> ep.JAXTensor:
    # img must be in channel_last format

    # modified according to https://github.com/google/jax/issues/862
    import jax.numpy as np

    img = x.raw

    resize_rates = (target_shape[1] / x.shape[1], target_shape[2] / x.shape[2])

    def interpolate_bilinear(  # type: ignore
        im: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        # based on http://stackoverflow.com/a/12729229
        col_lo = np.floor(cols).astype(int)
        col_hi = col_lo + 1
        row_lo = np.floor(rows).astype(int)
        row_hi = row_lo + 1

        def cclip(cols: np.ndarray) -> np.ndarray:  # type: ignore
            return np.clip(cols, 0, ncols - 1)

        def rclip(rows: np.ndarray) -> np.ndarray:  # type: ignore
            return np.clip(rows, 0, nrows - 1)

        nrows, ncols = im.shape[-3:-1]

        Ia = im[..., rclip(row_lo), cclip(col_lo), :]
        Ib = im[..., rclip(row_hi), cclip(col_lo), :]
        Ic = im[..., rclip(row_lo), cclip(col_hi), :]
        Id = im[..., rclip(row_hi), cclip(col_hi), :]

        wa = np.expand_dims((col_hi - cols) * (row_hi - rows), -1)
        wb = np.expand_dims((col_hi - cols) * (rows - row_lo), -1)
        wc = np.expand_dims((cols - col_lo) * (row_hi - rows), -1)
        wd = np.expand_dims((cols - col_lo) * (rows - row_lo), -1)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    nrows, ncols = img.shape[-3:-1]
    deltas = (0.5 / resize_rates[0], 0.5 / resize_rates[1])

    rows = np.linspace(deltas[0], nrows - deltas[0], np.int32(resize_rates[0] * nrows))
    cols = np.linspace(deltas[1], ncols - deltas[1], np.int32(resize_rates[1] * ncols))
    rows_grid, cols_grid = np.meshgrid(rows - 0.5, cols - 0.5, indexing="ij")

    img_resize_vec = interpolate_bilinear(img, rows_grid.flatten(), cols_grid.flatten())
    img_resize = img_resize_vec.reshape(
        img.shape[:-3] + (len(rows), len(cols)) + img.shape[-1:]
    )

    return ep.JAXTensor(img_resize)


def rescale_numpy(x: ep.NumPyTensor, target_shape: List[int]) -> ep.NumPyTensor:
    import numpy as np

    img = x.raw

    resize_rates = (target_shape[1] / x.shape[1], target_shape[2] / x.shape[2])

    def interpolate_bilinear(  # type: ignore
        im: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        # based on http://stackoverflow.com/a/12729229
        col_lo = np.floor(cols).astype(int)
        col_hi = col_lo + 1
        row_lo = np.floor(rows).astype(int)
        row_hi = row_lo + 1

        def cclip(cols: np.ndarray) -> np.ndarray:  # type: ignore
            return np.clip(cols, 0, ncols - 1)

        def rclip(rows: np.ndarray) -> np.ndarray:  # type: ignore
            return np.clip(rows, 0, nrows - 1)

        nrows, ncols = im.shape[-3:-1]

        Ia = im[..., rclip(row_lo), cclip(col_lo), :]
        Ib = im[..., rclip(row_hi), cclip(col_lo), :]
        Ic = im[..., rclip(row_lo), cclip(col_hi), :]
        Id = im[..., rclip(row_hi), cclip(col_hi), :]

        wa = np.expand_dims((col_hi - cols) * (row_hi - rows), -1)
        wb = np.expand_dims((col_hi - cols) * (rows - row_lo), -1)
        wc = np.expand_dims((cols - col_lo) * (row_hi - rows), -1)
        wd = np.expand_dims((cols - col_lo) * (rows - row_lo), -1)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    nrows, ncols = img.shape[-3:-1]
    deltas = (0.5 / resize_rates[0], 0.5 / resize_rates[1])

    rows = np.linspace(deltas[0], nrows - deltas[0], np.int32(resize_rates[0] * nrows))
    cols = np.linspace(deltas[1], ncols - deltas[1], np.int32(resize_rates[1] * ncols))
    rows_grid, cols_grid = np.meshgrid(rows - 0.5, cols - 0.5, indexing="ij")

    img_resize_vec = interpolate_bilinear(img, rows_grid.flatten(), cols_grid.flatten())
    img_resize = img_resize_vec.reshape(
        img.shape[:-3] + (len(rows), len(cols)) + img.shape[-1:]
    )

    return ep.NumPyTensor(img_resize)


def rescale_tensorflow(
    x: ep.TensorFlowTensor, target_shape: List[int]
) -> ep.TensorFlowTensor:
    import tensorflow

    img = x.raw

    img_resized = tensorflow.image.resize(img, size=target_shape[1:-1])

    return ep.TensorFlowTensor(img_resized)


def rescale_pytorch(x: ep.PyTorchTensor, target_shape: List[int]) -> ep.PyTorchTensor:
    import torch

    img = x.raw

    img_resized = torch.nn.functional.interpolate(
        img, size=target_shape[2:], mode="bilinear", align_corners=False
    )

    return ep.PyTorchTensor(img_resized)


def swap_axes(x: ep.TensorType, dim0: int, dim1: int) -> ep.TensorType:
    assert dim0 < x.ndim
    assert dim1 < x.ndim

    axes = list(range(x.ndim))
    axes[dim0] = dim1
    axes[dim1] = dim0

    return ep.transpose(x, tuple(axes))


def rescale_images(
    x: ep.TensorType, target_shape: Union[Tuple[int, ...], List[int]], channel_axis: int
) -> ep.TensorType:
    target_shape = list(target_shape)

    if channel_axis < 0:
        channel_axis = x.ndim - 1 + channel_axis

    if isinstance(x, ep.PyTorchTensor):
        if channel_axis != 1:
            x = swap_axes(x, channel_axis, 1)  # type: ignore

            target_shape[channel_axis], target_shape[1] = (
                target_shape[1],
                target_shape[channel_axis],
            )

        x = rescale_pytorch(x, target_shape)  # type: ignore

        if channel_axis != 1:
            x = swap_axes(x, channel_axis, 1)  # type: ignore

    elif isinstance(x, ep.TensorFlowTensor):
        if channel_axis != x.ndim - 1:
            x = swap_axes(x, channel_axis, x.ndim - 1)  # type: ignore

            target_shape[channel_axis], target_shape[x.ndim - 1] = (
                target_shape[x.ndim - 1],
                target_shape[channel_axis],
            )

        x = rescale_tensorflow(x, target_shape)  # type: ignore

        if channel_axis != x.ndim - 1:
            x = swap_axes(x, channel_axis, x.ndim - 1)  # type: ignore

    elif isinstance(x, ep.NumPyTensor):
        if channel_axis != x.ndim - 1:
            x = swap_axes(x, channel_axis, x.ndim - 1)  # type: ignore

            target_shape[channel_axis], target_shape[x.ndim - 1] = (
                target_shape[x.ndim - 1],
                target_shape[channel_axis],
            )

        x = rescale_numpy(x, target_shape)  # type: ignore
        if channel_axis != x.ndim - 1:
            x = swap_axes(x, channel_axis, x.ndim - 1)  # type: ignore

    elif isinstance(x, ep.JAXTensor):
        if channel_axis != x.ndim - 1:
            x = swap_axes(x, channel_axis, x.ndim - 1)  # type: ignore

            target_shape[channel_axis], target_shape[x.ndim - 1] = (
                target_shape[x.ndim - 1],
                target_shape[channel_axis],
            )

        x = rescale_jax(x, target_shape)  # type: ignore
        if channel_axis != x.ndim - 1:
            x = swap_axes(x, channel_axis, x.ndim - 1)  # type: ignore

    return x
