import os

import numpy as np


def softmax(logits):
    """Transforms predictions into probability values.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.

    Returns
    -------
    `numpy.ndarray`
        Probability values corresponding to the logits.
    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def crossentropy(label, logits):
    """Calculates the cross-entropy.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.
    label : int
        The label describing the target distribution.

    Returns
    -------
    float
        The cross-entropy between softmax(logits) and onehot(label).

    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = np.sum(e)
    ce = np.log(s) - logits[label]
    return ce


def batch_crossentropy(label, logits):
    """Calculates the cross-entropy for a batch of logits.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model for a batch of inputs.
    label : int
        The label describing the target distribution.

    Returns
    -------
    np.ndarray
        The cross-entropy between softmax(logits[i]) and onehot(label)
        for all i.

    """

    assert logits.ndim == 2

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    s = np.sum(e, axis=1)
    ces = np.log(s) - logits[:, label]
    return ces


def binarize(x, values, threshold=None, included_in='upper'):
    """Binarizes the values of x.

    Parameters
    ----------
    values : tuple of two floats
        The lower and upper value to which the inputs are mapped.
    threshold : float
        The threshold; defaults to (values[0] + values[1]) / 2 if None.
    included_in : str
        Whether the threshold value itself belongs to the lower or
        upper interval.

    """
    lower, upper = values

    if threshold is None:
        threshold = (lower + upper) / 2.

    x = x.copy()
    if included_in == 'lower':
        x[x <= threshold] = lower
        x[x > threshold] = upper
    elif included_in == 'upper':
        x[x < threshold] = lower
        x[x >= threshold] = upper
    else:
        raise ValueError('included_in must be "lower" or "upper"')
    return x


def imagenet_example(shape=(224, 224), data_format='channels_last'):
    """ Returns an example image and its imagenet class label.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    data_format : str
        "channels_first" or "channels_last"

    Returns
    -------
    image : array_like
        The example image.

    label : int
        The imagenet label associated with the image.

    """
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']

    from PIL import Image
    path = os.path.join(os.path.dirname(__file__), 'example.png')
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image, 282


def onehot_like(a, index, value=1):
    """Creates an array like a, with all values
    set to 0 except one.

    Parameters
    ----------
    a : array_like
        The returned one-hot array will have the same shape
        and dtype as this array
    index : int
        The index that should be set to `value`
    value : single value compatible with a.dtype
        The value to set at the given index

    Returns
    -------
    `numpy.ndarray`
        One-hot array with the given value at the given
        location and zeros everywhere else.

    """

    x = np.zeros_like(a)
    x[index] = value
    return x
