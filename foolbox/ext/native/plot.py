import eagerpy as ep
import numpy as np
import matplotlib.pyplot as plt


def images(
    images,
    *,
    n=None,
    data_format=None,
    bounds=(0, 1),
    ncols=None,
    nrows=None,
    figsize=None,
    scale=1,
    **kwargs,
):
    x = ep.astensor(images)
    assert x.ndim == 4
    if n is not None:
        x = x[:n]
    if data_format is None:
        channels_first = x.shape[1] == 1 or x.shape[1] == 3
        channels_last = x.shape[-1] == 1 or x.shape[-1] == 3
        if channels_first == channels_last:
            raise ValueError("data_format ambigous, please specify explicitly")
    else:
        channels_first = data_format == "channels_first"
        channels_last = data_format == "channels_last"
    assert channels_first != channels_last
    x = x.numpy()
    if channels_first:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    min_, max_ = bounds
    x = (x - min_) / (max_ - min_)

    if nrows is None and ncols is None:
        nrows = 1
    if ncols is None:
        ncols = (len(x) + nrows - 1) // nrows
    elif nrows is None:
        nrows = (len(x) + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * scale, nrows * scale)
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
        **kwargs,
    )

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            i = row * ncols + col
            if i < len(x):
                ax.imshow(x[i])
