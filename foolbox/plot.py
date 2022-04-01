from typing import Tuple, Any, Optional
import numpy as np
import eagerpy as ep


def images(
    images: Any,
    *,
    n: Optional[int] = None,
    data_format: Optional[str] = None,
    bounds: Tuple[float, float] = (0, 1),
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    scale: float = 1,
    **kwargs: Any,
) -> None:
    import matplotlib.pyplot as plt

    x: ep.Tensor = ep.astensor(images)
    if x.ndim != 4:
        raise ValueError(
            "expected images to have four dimensions: (N, C, H, W) or (N, H, W, C)"
        )
    if n is not None:
        x = x[:n]
    if data_format is None:
        channels_first = x.shape[1] == 1 or x.shape[1] == 3
        channels_last = x.shape[-1] == 1 or x.shape[-1] == 3
        if channels_first == channels_last:
            raise ValueError("data_format ambigous, please specify it explicitly")
    else:
        channels_first = data_format == "channels_first"
        channels_last = data_format == "channels_last"
        if not channels_first and not channels_last:
            raise ValueError(
                "expected data_format to be 'channels_first' or 'channels_last'"
            )
    assert channels_first != channels_last
    x = x.numpy()
    if channels_first:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    min_, max_ = bounds
    x = (x - min_) / (max_ - min_)

    if nrows is None and ncols is None:
        nrows = 1
    if ncols is None:
        assert nrows is not None
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
                if x.shape[-1] == 1:
                    ax.imshow(x[i][:, :, 0])
                else:
                    ax.imshow(x[i])
