import logging

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from .base import Attack
from .base import call_decorator


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
    """Adversarial attack that distorts the image, i.e. changes the locations
    of pixels. The algorithm is described in [1]_,
    a Repository with the original code can be found in [2]_.
    References
    ----------
    .. [1]_ Rima Alaifari, Giovanni S. Alberti, and Tandri Gauksson:
            "ADef: an Iterative Algorithm to Construct Adversarial
            Deformations", https://arxiv.org/abs/1804.07729
    .. [2]_ https://gitlab.math.ethz.ch/tandrig/ADef/tree/master
    """

    def _initialize(self):
        self.vector_field = None

    @call_decorator
    def __call__(self, input_or_adv, unpack=True, max_iter=100,
                 max_norm=np.inf, label=None, smooth=1.0, subsample=10):

        """Parameters
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
        max_norm : float
            Maximum l2 norm of vector field (default max_norm = numpy.inf).
        smooth : float >= 0
            Width of the Gaussian kernel used for smoothing.
            (default is smooth = 0 for no smoothing).
        subsample : int >= 2
            Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much
            faster. (default subsample = 10)
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            return

        perturbed = a.unperturbed.copy()  # is updated in every iteration

        # image_original is not updated, but kept as a copy
        image_original = a.unperturbed.copy()
        target_class = a.target_class()
        targeted = target_class is not None
        original_label = a.original_class

        # ADef targets classes according to their prediction score. If the
        # attack is untargeted, ADef will take the labels of the top
        # 'subsample' classes without the top class. The top class is
        # the class with the highest probability and not among the targets.
        # Using a 'subsample' of classes is faster than taking all the
        # remaining 999 classes of ImageNet into account. For a targeted
        # attack, it is necessary to find the probability of the target
        # class and pass this index to ind_of_candidates (not the actual
        # target).
        if targeted:
            logits, _ = a.forward_one(perturbed)
            pred_sorted = (-logits).argsort()
            index_of_target_class, = np.where(pred_sorted == target_class)
            ind_of_candidates = index_of_target_class
        else:
            # choose the top-k classes
            logging.info('Only testing the top-{} classes'.format(subsample))
            assert isinstance(subsample, int)
            assert subsample >= 2
            ind_of_candidates = np.arange(1, subsample)

        # Number of classes to target
        num_classes = ind_of_candidates.size

        n = 0  # iteration number

        color_axis = a.channel_axis(batch=False)  # get color axis
        assert color_axis in [0, 2]
        hw = [perturbed.shape[i] for i in range(perturbed.ndim)
              if i != color_axis]
        h, w = hw

        logits, is_adv = a.forward_one(perturbed)

        # Indices of the 'num_classes' highest values in descending order:
        candidates = np.argsort(-logits)[ind_of_candidates]

        # fx[lab] is negative if the model prefers the original label
        # for x over the label 'lab'.
        fx = logits - logits[original_label]

        norm_full = 0  # norm of the vector field
        vec_field_full = np.zeros((h, w, 2))  # the vector field

        current_label = original_label
        logging.info('Iterations finished: 0')
        logging.info('Current label: {} '.format(current_label))

        for step in range(max_iter):
            n += 1
            _, is_adv = a.forward_one(perturbed)
            if is_adv:
                a.forward_one(perturbed)
                logging.info(
                    'Image successfully deformed from {} to {}'.format(
                        original_label, current_label))
                self.vector_field = vec_field_full
                return

            d1x, d2x = _difference_map(perturbed, color_axis)

            logits_for_grad = np.zeros_like(logits)
            logits_for_grad[original_label] = 1

            grad_original = a.backward_one(logits_for_grad, perturbed)

            # Find vector fields for the image and each candidate label.
            # Keep the smallest vector field for each image.
            norm_min = np.inf

            # iterate over all candidate classes
            for target_no in range(num_classes):

                target_label = candidates[target_no]
                logits_for_grad = np.zeros_like(logits)
                logits_for_grad[target_label] = 1

                # gradient of the target label w.r.t. image
                grad_target = a.backward_one(logits_for_grad, perturbed)

                # Derivative of the binary classifier 'F_lab - F_orig'
                dfx = grad_target - grad_original
                f_im = fx[target_label]

                # create the vector field
                vec_field_target = _create_vec_field(
                    f_im, dfx, d1x, d2x, color_axis, smooth)

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
            logits, _ = a.forward_one(perturbed)
            current_label = np.argmax(logits)
            fx = logits - logits[current_label]

            logging.info('Iterations finished: {} '.format(n))
            logging.info('Current label: {} '.format(current_label))
            logging.info('Norm vector field: {} '.format(norm_full))

        logits, _ = a.forward_one(perturbed)
        current_label = np.argmax(logits)
        logging.info('{} -> {}'.format(original_label, current_label))

        a.forward_one(perturbed)

        self.vector_field = vec_field_full
        return
