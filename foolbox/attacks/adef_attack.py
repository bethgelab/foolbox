import logging

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from .base import Attack
from .base import call_decorator
from ..criteria import Misclassification


def _transpose_image(image):
    # transpose the image so the color axis
    # is at the front: image.shape is then c x h x w:
    return np.transpose(image, (2, 0, 1))


def _re_transpose_image(image):
    # transpose the image back so the color axis
    # is at the end: image.shape is then h x w x c:
    return np.transpose(image, (1, 2, 0))


def _difference_map(image, color_axis):
    """Difference map of the image.
    Approximate derivatives of the function image[c, :, :]
    (e.g. PyTorch) or image[:, :, c] (e.g. Keras).

    dfdx, dfdy = difference_map(image)

    In:
    image: numpy.ndarray
        of shape C x h x w or h x w x C, with C = 1 or C = 3
        (color channels), h, w >= 3, and [type] is 'Float' or
        'Double'. Contains the values of functions f_b:
        R ^ 2 -> R ^ C, b = 1, ..., B, on the grid
        {0, ..., h - 1} x {0, ..., w - 1}.

    Out:
    dfdx: numpy.ndarray
    dfdy: numpy.ndarray
        of shape C x h x w or h x w x C contain the x and
        y derivatives of f at the points on the grid,
        approximated by central differences (except on
        boundaries):
        For c = 0, ... , C, i = 1, ..., h - 2,
        j = 1, ..., w - 2.

        e.g. for shape = c x h x w:
        dfdx[c, i, j] = (image[c, i, j + 1] -
            image[c, i, j - 1]) / 2
        dfdx[c, i, j] = (image[c, i + 1, j] -
            image[c, i - 1, j]) / 2

    positive x-direction is along rows from left to right.
    positive y-direction is along columns from above to below.

    """

    if color_axis == 2:
        image = _transpose_image(image)
    # Derivative in x direction (rows from left to right)
    dfdx = np.zeros_like(image)
    # forward difference in first column
    dfdx[:, :, 0] = image[:, :, 1] - image[:, :, 0]
    # backwards difference in last column
    dfdx[:, :, -1] = image[:, :, -1] - image[:, :, -2]
    # central difference elsewhere
    dfdx[:, :, 1:-1] = 0.5 * (image[:, :, 2:] - image[:, :, :-2])

    # Derivative in y direction (columns from above to below)
    dfdy = np.zeros_like(image)
    # forward difference in first row
    dfdy[:, 0, :] = image[:, 1, :] - image[:, 0, :]
    # backwards difference in last row
    dfdy[:, -1, :] = image[:, -1, :] - image[:, -2, :]
    # central difference elsewhere
    dfdy[:, 1:-1, :] = 0.5 * (image[:, 2:, :] - image[:, :-2, :])

    return dfdx, dfdy


def _compose(image, vec_field, color_axis):
    """Calculate the composition of the function image with the vector
    field vec_field by interpolation.

    new_func = compose(image, vec_field)

    In:
    image: numpy.ndarray
        of shape C x h x w with C = 3 or C = 1 (color channels),
        h, w >= 2, and [type] = 'Float' or 'Double'.
        Contains the values of a function f: R ^ 2 -> R ^ C
        on the grid {0, ..., h - 1} x {0, ..., w - 1}.
    vec_field: numpy.array
        of shape (h, w, 2)

    vec_field[y, x, 0] is the x-coordinate of the vector vec_field[y, x]
    vec_field[y, x, 1] is the y-coordinate of the vector vec_field[y, x]

    positive x-direction is along rows from left to right
    positive y-direction is along columns from above to below

    """

    if color_axis == 2:
        image = _transpose_image(image)

    c, h, w = image.shape  # colors, height, width
    hrange = np.arange(h)
    wrange = np.arange(w)
    MGx, MGy = np.meshgrid(wrange, hrange)

    defMGx = (MGx + vec_field[:, :, 0]).clip(0, w - 1)
    defMGy = (MGy + vec_field[:, :, 1]).clip(0, h - 1)

    new_image = np.empty_like(image)

    for channel in range(c):
        # Get a linear interpolation for this color channel.
        interpolation = RectBivariateSpline(hrange, wrange, image[channel],
                                            kx=1, ky=1)

        # grid = False since the deformed grid is irregular
        new_image[channel] = interpolation(defMGy, defMGx, grid=False)
    if color_axis == 2:
        return _re_transpose_image(new_image)
    else:
        return new_image


def _create_vec_field(fval, gradf, d1x, d2x, color_axis, smooth=0):
    """Calculate the deformation vector field

    In:
    fval: float
    gradf: numpy.ndarray
        of shape C x h x w with C = 3 or C = 1
        (color channels), h, w >= 1.
    d1x: numpy.ndarray
        of shape C x h x w and [type] = 'Float' or 'Double'.
    d2x: numpy.ndarray
        of shape C x h x w and [type] = 'Float' or 'Double'.
    smooth: float
        Width of the Gaussian kernel used for smoothing
        (default is 0 for no smoothing).

    Out:
    vec_field: numpy.ndarray
        of shape (2, h, w).

    """

    if color_axis == 2:
        gradf = _transpose_image(gradf)

    c, h, w = gradf.shape  # colors, height, width

    # Sum over color channels
    alpha1 = np.sum(gradf * d1x, axis=0)
    alpha2 = np.sum(gradf * d2x, axis=0)

    norm_squared_alpha = (alpha1 ** 2).sum() + (alpha2 ** 2).sum()

    # Smoothing
    if smooth > 0:
        alpha1 = gaussian_filter(alpha1, smooth)
        alpha2 = gaussian_filter(alpha2, smooth)
        norm_squared_alpha = (alpha1 ** 2).sum() + (alpha2 ** 2).sum()
        # In theory, we need to apply the filter a second time.
        alpha1 = gaussian_filter(alpha1, smooth)
        alpha2 = gaussian_filter(alpha2, smooth)

    vec_field = np.empty((h, w, 2))
    vec_field[:, :, 0] = -fval * alpha1 / norm_squared_alpha
    vec_field[:, :, 1] = -fval * alpha2 / norm_squared_alpha

    return vec_field


class ADefAttack(Attack):
    """Adversarial Attack that distorts the image, i.e. changes the locations
    of pixels. The algorithm is described in [1],
    A Repository with the original code can be found in [2].

    References
    ----------
    .. [1] Rima Alaifari, Giovanni S. Alberti, and Tandri Gauksson1:
           "ADef: an Iterative Algorithm to Construct Adversarial
           Deformations", https://arxiv.org/pdf/1804.07729.pdf

    .. [2] https://gitlab.math.ethz.ch/tandrig/ADef/tree/master

    Parameters
    ----------
    input_or_adv : `numpy.ndarray` or :class:`Adversarial`
        The original, unperturbed input as a `numpy.ndarray` or
        an :class:`Adversarial` instance.
    label : int
        The reference label of the original input. Must be passed
        if `a` is a `numpy.ndarray`, must not be passed if `a` is
        an :class:`Adversarial` instance.
    unpack : bool
        If true, returns the adversarial input, otherwise returns
        the Adversarial object.
    max_iter : int > 0
        Maximum number of iterations (default max_iter = 100).
    ind_of_candidates : int > 0, or array_like of int values > 0
        The indices of labels to target in the ordering of descending
        confidence.
        For example:
        - ind_of_candidates = [1, 2, 3] to target the top three labels.
        - ind_of_candidates = 5 to to target the fifth best label.
    max_norm : float
        Maximum l2 norm of vector field (default max_norm = numpy.inf).
    overshoot : float >= 1
        Multiply the resulting vector field by this number,
        if deformed image is still correctly classified
        (default is overshoot = 1 for no overshooting).
    smooth : float >= 0
        Width of the Gaussian kernel used for smoothing.
        (default is smooth = 0 for no smoothing).
    targeting : bool
        targeting = False (default) to stop as soon as model misclassifies
        input targeting = True to stop only once a candidate label is achieved.

    """

    def __init__(self, model=None, criterion=Misclassification()):
        super(ADefAttack, self).__init__(model=model, criterion=criterion)
        self.vector_field = None

    @call_decorator
    def __call__(self, input_or_adv, ind_of_candidates=1, unpack=True,
                 max_iter=100, max_norm=np.inf, label=None,
                 overshoot=1.1, smooth=1.0, targeting=False):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        # Include the correct label (index 0) in the list of targets.
        # Remove duplicates and sort the label indices.
        ind_of_candidates = np.unique(np.append(ind_of_candidates, 0))
        # Remove negative entries.
        ind_of_candidates = ind_of_candidates[ind_of_candidates >= 0]

        # Number of classes to target + 1 ( >= 2).
        # Example 1: num_classes = 10 for MNIST
        # Example 2: If ind_of_candidates == 1,
        # then only the second highest label is targeted
        num_classes = ind_of_candidates.size

        n = 0  # iteration number

        perturbed = a.original_image.copy()  # is updated in every iteration

        image_original = a.original_image.copy()  # is not updated,
        # but kept as a copy

        color_axis = a.channel_axis(batch=False)  # get color axis
        assert color_axis in [0, 2]
        hw = [perturbed.shape[i] for i in range(perturbed.ndim)
              if i != color_axis]
        h, w = hw

        logits, grad, is_adv = a.predictions_and_gradient(perturbed)

        logits = np.expand_dims(logits, axis=0)

        # Indices of the 'num_classes' highest values in descending order:
        candidates = np.argsort(-logits)[:, ind_of_candidates]
        original_label = candidates[:, 0]

        # fx[lab] is negative if the model prefers the original label
        # for x over the label 'lab'.
        fx = (logits.transpose() - logits[0, original_label]).transpose()

        norm_full = 0  # norm of the vector field
        vec_field_full = np.zeros((h, w, 2))  # the vector field

        current_label = original_label
        logging.info('Iterations finished: 0')
        logging.info('Current labels: {} '.format(current_label))

        for step in range(max_iter):
            n += 1
            logits, grad, is_adv = a.predictions_and_gradient(perturbed)
            if is_adv:
                a.predictions(perturbed)
                self.vector_field = vec_field_full
                return

            d1x, d2x = _difference_map(perturbed, color_axis)

            logits_for_grad = np.zeros_like(logits)
            logits_for_grad[original_label[0]] = 1

            grad_original = a.backward(logits_for_grad, perturbed)

            # Find vector fields for the image and each candidate label.
            # Keep the smallest vector field for each image.
            norm_min = np.inf

            # iterate over all candidate classes
            for target_no in range(1, num_classes):

                target_labels = candidates[0, target_no]
                logits_for_grad = np.zeros_like(logits)
                logits_for_grad[target_labels] = 1

                # gradient of the target label w.r.t. image
                grad_target = a.backward(logits_for_grad, perturbed)

                # Derivative of the binary classifier 'F_lab - F_orig'
                dfx = grad_target - grad_original

                f_im = fx[0, candidates[0, target_no]]

                # create the vector field
                vec_field_target = _create_vec_field(
                    f_im, dfx, d1x, d2x, color_axis, smooth
                )

                vec_field_target += vec_field_full

                # l2 norm of vector field.
                norm_target = np.linalg.norm(vec_field_target.ravel())

                # choose the vector field with the smallest norm
                if norm_target < norm_min:
                    norm_min = norm_target
                    vec_field_min = vec_field_target

            # Update the image by applying the vector field,
            # the vector field is always applied to the original image,
            # since the current vector field is added to all prior
            # vector fields via vec_field_target += vec_field_full
            perturbed = _compose(image_original.copy(), vec_field_min,
                                 color_axis)

            vec_field_full = vec_field_min
            norm_full = norm_min

            # getting the current label after applying the vector field
            fx, _ = a.predictions(perturbed)
            fx = np.expand_dims(fx, axis=0)
            current_label = np.argmax(fx, axis=1)
            fx = (fx.transpose() - fx[0, current_label]).transpose()

            # See if we have been successful.
            if targeting and (current_label in candidates[0, 1:]):
                logging.info(
                    'Image successfully deformed from {} to {}'.format(
                        original_label, current_label))
                continue
            elif (not targeting) and current_label != original_label:
                logging.info(
                    'Image successfully deformed from {} to {}'.format(
                        original_label, current_label))
            logging.info('Iterations finished: {} '.format(n))
            logging.info('Current labels: {} '.format(current_label))
            logging.info('Norm vector field: {} '.format(norm_full))

        # Overshooting
        try_overshoot = False
        have_highest_confidence = (fx == 0)
        predicted_labels = np.where(have_highest_confidence)[0]

        if len(predicted_labels) > 1:
            logging.info('Deformed image lies on the decision '
                         'boundary of labels: {} '.format(predicted_labels))
            try_overshoot = True

        # Overshoot if deformed image is still correctly classified,
        # lies on a decision boundary or, in case of targeting,
        # is not classified as a candidate label.
        try_overshoot = try_overshoot or (original_label in predicted_labels)
        try_overshoot = try_overshoot or (targeting and not
                                          (current_label in candidates[0, 1:]))
        try_overshoot = try_overshoot and (overshoot > 1) and (norm_full > 0)

        if try_overshoot:
            os = min(overshoot, max_norm / norm_full)
            logging.info('Overshooting: vec_field ->'
                         ' {} * vec_field '.format(os))

            vec_field_full = os * vec_field_full

            perturbed = _compose(image_original.copy(), vec_field_full,
                                 color_axis)

        fx, _ = a.predictions(perturbed)
        fx = np.expand_dims(fx, axis=0)
        current_label = np.argmax(fx, axis=1)
        logging.info('{} -> {}'.format(original_label, current_label))

        a.predictions(perturbed)

        self.vector_field = vec_field_full
        return
