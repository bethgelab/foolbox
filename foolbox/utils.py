import os
import warnings

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


def binarize(x, values, threshold=None, included_in="upper"):
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
        threshold = (lower + upper) / 2.0

    x = x.copy()
    if included_in == "lower":
        x[x <= threshold] = lower
        x[x > threshold] = upper
    elif included_in == "upper":
        x[x < threshold] = lower
        x[x >= threshold] = upper
    else:
        raise ValueError('included_in must be "lower" or "upper"')
    return x


def imagenet_example(shape=(224, 224), data_format="channels_last", bounds=(0, 255)):
    """ Returns an example image and its imagenet class label.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned image.
    data_format : str
        "channels_first" or "channels_last"
    bounds : tuple
        smallest and largest allowed pixel value

    Returns
    -------
    image : array_like
        The example image.

    label : int
        The imagenet label associated with the image.

    NOTE: This function is deprecated and will be removed in the future.
    """
    assert len(shape) == 2
    assert data_format in ["channels_first", "channels_last"]

    from PIL import Image

    path = os.path.join(os.path.dirname(__file__), "example.png")
    image = Image.open(path)
    image = image.resize(shape)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == "channels_first":
        image = np.transpose(image, (2, 0, 1))
    if bounds != (0, 255):
        image = image / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return image, 282


def samples(
    dataset="imagenet",
    index=0,
    batchsize=1,
    shape=(224, 224),
    data_format="channels_last",
    bounds=(0, 255),
):
    """ Returns a batch of example images and the corresponding labels

    Parameters
    ----------
    dataset : string
        The data set to load (options: imagenet, mnist, cifar10,
        cifar100, fashionMNIST)
    index : int
        For each data set 20 example images exist. The returned batch
        contains the images with index [index, index + 1, index + 2, ...]
    batchsize : int
        Size of batch.
    shape : list of integers
        The shape of the returned image (only relevant for Imagenet).
    data_format : str
        "channels_first" or "channels_last"
    bounds : tuple
        smallest and largest allowed pixel value

    Returns
    -------
    images : array_like
        The batch of example images

    labels : array of int
        The labels associated with the images.

    """
    from PIL import Image

    images, labels = [], []
    basepath = os.path.dirname(__file__)
    samplepath = os.path.join(basepath, "data")
    files = os.listdir(samplepath)

    if batchsize > 20:
        warnings.warn(
            "foolbox.utils.samples() has only 20 samples and repeats itself if batchsize > 20"
        )

    for idx in range(index, index + batchsize):
        i = idx % 20

        # get filename and label
        file = [n for n in files if "{}_{:02d}_".format(dataset, i) in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)

        if dataset == "imagenet":
            image = image.resize(shape)

        image = np.asarray(image, dtype=np.float32)

        if dataset != "mnist" and data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        images.append(image)
        labels.append(label)

    images = np.stack(images)
    labels = np.array(labels)

    if bounds != (0, 255):
        images = images / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return images, labels


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
