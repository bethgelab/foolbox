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
