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


def imagenet_example(shape=(224, 224)):
    """ Returns an example image and its imagenet class label.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.

    Returns
    -------
    image : array_like
        The example image.

    label : int
        The imagenet label associated with the image.

    """
    import scipy
    path = os.path.join(os.path.dirname(__file__), 'example.png')
    image = scipy.misc.imread(path)[:, :, :3]
    return scipy.misc.imresize(image, shape).astype(np.float32), 282
