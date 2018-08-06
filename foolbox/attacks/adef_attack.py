from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from .base import Attack
from .base import call_decorator
from ..criteria import Misclassification


def _preprocess_image(image, preprocessing, color_axis):
    if color_axis == 2:
        return image
        image = _transpose_image(image) / 255
    psub, pdiv = preprocessing
    psub = np.asarray(psub, dtype=image.dtype)
    pdiv = np.asarray(pdiv, dtype=image.dtype)
    result = image
    if np.any(psub != 0):
        result = image - psub  # creates a copy
    if np.any(pdiv != 1):
        if np.any(psub != 0):  # already copied
            result /= pdiv  # in-place
        else:
            result = result / pdiv  # creates a copy
    assert result.dtype == image.dtype
    if color_axis == 2:
        return _re_transpose_image(result)
    else:
        return result


def _deprocess_image(image, preprocessing, color_axis):
    if color_axis == 2:
        return image
        image = _transpose_image(image)
    psub, pdiv = preprocessing
    psub = np.asarray(psub, dtype=image.dtype)
    pdiv = np.asarray(pdiv, dtype=image.dtype)
    result = image
    if np.any(pdiv != 1):
        if np.any(psub != 0):  # already copied
            result *= pdiv  # in-place
        else:
            result = result * pdiv  # creates a copy
    if np.any(psub != 0):
        result = image + psub  # creates a copy
    assert result.dtype == image.dtype
    if color_axis == 2:
        return _re_transpose_image(result) * 255
    else:
        return result


def _deprocess_gradient(gradient, preprocessing, color_axis):
    if color_axis == 2:
        return gradient
        gradient = _transpose_image(gradient)
    _, pdiv = preprocessing
    pdiv = np.asarray(pdiv, dtype=gradient.dtype)
    if np.any(pdiv != 1):
        result = gradient * pdiv
    else:
        result = gradient
    assert result.dtype == gradient.dtype
    if color_axis == 2:
        return _re_transpose_image(result)
    else:
        return result


def _transpose_image(image):
    # transpose the image so the color axis
    # is at the front: image.shape is then cxhxw:
    return np.transpose(image, (2, 0, 1))


def _re_transpose_image(image):
    # transpose the image back so the color axis
    #  is at the end: image.shape is then hxwxc:
    return np.transpose(image, (1, 2, 0))


def _difference_map(image, color_axis):
    """
    Difference map of the image
    Approximate derivatives of the function image[c,:,:]
    (e.g. PyTorch) or image[:,:,c] (e.g. Keras).

    dfdx, dfdy = difference_map( image )

    In:
    image: numpy.ndarray
        of shape Cxhxw or hxwxC, with C = 1 or C = 3 (color channels),
        h,w >= 3, and [type] is 'Float' or 'Double'.
        Contains the values of functions f_b: R^2 -> R^C, b=1,...,B,
        on the grid {0,...,h-1}x{0,...,w-1}.

    Out:
    dfdx: numpy.ndarray
    dfdy: numpy.ndarray
        of shape Cxhxw or hxwxC contain the x and y derivatives of f
        at the points on the grid, approximated by central differences
        (except on boundaries):
        For c=0,...,C, i=1,...,h-2, j=1,...,w-2

        e.g. for shape = cxhxw:
        dfdx[c,i,j] = (image[c,i,j+1] - image[c,i,j-1])/2
        dfdx[c,i,j] = (image[c,i+1,j] - image[c,i-1,j])/2

    positive x-direction is along rows from left to right.
    positive y-direction is along columns from above to below.

    #################################################
    Note from Evgenia: I am not sure why they did not
    use the function np.gradient for this.
    It seems to give very similar results. I have
    let it at the original implementation, though.
    #################################################

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
    """
    Calculate the composition of the function image with the vector
    field vec_field by interpolation.

    new_func = compose(  image, vec_field )

    In:
    image: numpy.ndarray
        of shape Cxhxw with C=3 or C=1 (color channels), h,w >= 2,
        and [type] = 'Float' or 'Double'.
        Contains the values of a function f:R^2 -> R^C
        on the grid {0,...,h-1}x{0,...,w-1}.
    vec_field: numpy.array
        of shape (h,w,2)

    vec_field[y,x,0] is the x-coordinate of the vector vec_field[y,x]
    vec_field[y,x,1] is the y-coordinate of the vector vec_field[y,x]

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

        # grid=False since the deformed grid is irregular
        new_image[channel] = interpolation(defMGy, defMGx, grid=False)
    if color_axis == 2:
        return _re_transpose_image(new_image)
    else:
        return new_image


def _create_tau(fval, gradf, d1x, d2x, color_axis, smooth=0):
    """
    tau = create_tau( fval, gradf, d1x, d2x )

    In:
    fval: float
    gradf: numpy.ndarray
        of shape Cxhxw with C=3 or C=1 (color channels), h,w >= 1.
    d1x: numpy.ndarray
        of shape Cxhxw and [type] = 'Float' or 'Double'.
    d2x: numpy.ndarray
        of shape Cxhxw and [type] = 'Float' or 'Double'.
    smooth: float
        Width of the Gaussian kernel used for smoothing
        (default is 0 for no smoothing).

    Out:
    tau: numpy.ndarray
        of shape (2,h,w).
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

    tau = np.empty((h, w, 2))
    tau[:, :, 0] = -fval * alpha1 / norm_squared_alpha
    tau[:, :, 1] = -fval * alpha2 / norm_squared_alpha

    return tau


class ADefAttack(Attack):
    """
    Adversarial Attack that distorts the image, i.e. changes the locations
     of pixels. The algorithm is described in [1],
    A Repository with the original code can be found in [2].

     Find an adversarial deformation of each image in batch w.r.t model.

        Parameters  (from FoolBox)
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
    max_iter: int > 0
        Maximum number of iterations (default max_iter = 100).
                     (from ADef)
    ind_of_candidates: int > 0, or array_like of int values > 0
        The indices of labels to target in the ordering of descending
        confidence.
        For example:
        - ind_of_candidates = [1,2,3] to target the top three labels.
        - ind_of_candidates = 5 to to target the fifth best label.
    max_norm: float
        Maximum l2 norm of vector field (default max_norm = numpy.inf).
    overshoot: float >= 1
        Multiply the resulting vector field by this number,
        if deformed image is still correctly classified
        (default is overshoot = 1 for no overshooting).
    smooth: float >= 0
        Width of the Gaussian kernel used for smoothing.
        (default is smooth = 0 for no smoothing).
    targeting: bool
        targeting = False (default) to stop as soon as model misclassifies
        input targeting = True to stop only once a candidate label is achieved.
    verbose: bool
        verbose = True (default) to print progress,
        verbose = False for silence.

    Out:

    To access the vectorfields, an additional attribute
    "result_data" (type=dict) is created 'on the fly' after the image
    is detected as being "adversarial" and before "return". It can be accessed
    from the main program via attack.result_data. Currently, it is not
    initialized before it is created (which is maybe not very nice),
    because I did not know whether this is actually desired,
    since it is only required and thus created for this particular
    attack. Also, I was not sure where to initialize it.

    result_data: dict
        result_data['vector_fields']: numpy.ndarray
            of shape hxwx2.
            The deforming vector fields.
        result_data['deformed_label']: numpy.ndarray
            The predicted label of the deformed image.

      References
    ----------
      [1] Rima Alaifari, Giovanni S. Alberti, and Tandri Gauksson1:
           "ADef: an Iterative Algorithm to Construct Adversarial
           Deformations", https://arxiv.org/pdf/1804.07729.pdf

      [2] https://gitlab.math.ethz.ch/tandrig/ADef/tree/master

    """

    def __init__(self, model=None, criterion=Misclassification()):
        super(ADefAttack, self).__init__(model=model, criterion=criterion)
        self.vector_field = None

    @call_decorator
    def __call__(self, input_or_adv, ind_of_candidates=1, unpack=True,
                 max_iter=100, max_norm=np.inf, label=None,
                 overshoot=1.1, smooth=1., targeting=False, verbose=False,
                 preprocessing=(0, 1)):

        vprint = print if verbose else lambda *a, **k: None
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

        # Number of classes to target + 1 (>=2).
        #  Example 1: num_classes = 10 for MNIST
        # Example 2: If ind_of_candidates == 1,
        # then only the second highest label is targeted
        num_classes = ind_of_candidates.size

        n = 0  # iteration number

        perturbed = a.original_image.copy()  # is updated in every iteration

        image_original = perturbed.copy()  # is not updated, but kept as a copy

        color_axis = a.channel_axis(batch=False)  # get color axis
        assert color_axis in [0, 2]
        hw = [perturbed.shape[i] for i in range(perturbed.ndim)
              if i != color_axis]
        h, w = hw

        min_, max_ = a.bounds()
        logits, grad, is_adv = a.predictions_and_gradient(perturbed)

        logits = np.expand_dims(logits, axis=0)

        # Indices of the 'num_classes' highest values in descending order:
        candidates = np.argsort(-logits)[:, ind_of_candidates]
        original_label = candidates[:, 0]

        # fx[lab] is negative if the model prefers the original label
        # for x over the label 'lab'.
        fx = (logits.transpose() - logits[0, original_label]).transpose()

        norm_full = 0  # norm of the vector field
        tau_full = np.zeros((h, w, 2))  # the vector field

        current_label = original_label
        vprint('Iterations finished: 0')
        vprint('\tCurrent labels: ' + str(current_label))

        for step in range(max_iter):
            n += 1
            logits, grad, is_adv = a.predictions_and_gradient(perturbed)
            if is_adv:
                a.predictions(perturbed)
                self.vector_field = tau_full
                return

            # Calculate the difference map for the image
            # get_preprocessed, because: the image has to be preprocessed
            # in a special way for PyTorch. In FoolBox, this is an
            # internal private method that is only used when e.g.
            # the gradients w.r.t. the image are
            # calculated. Here, we need this preprocessed image explicitly
            # to calculate the difference map.
            d1x, d2x = _difference_map(
                _preprocess_image(perturbed, preprocessing, color_axis),
                color_axis)

            logits_for_grad = np.zeros_like(logits)
            logits_for_grad[original_label[0]] = 1
            grad_original = _deprocess_gradient(
                a.backward(logits_for_grad, perturbed),
                preprocessing, color_axis)

            # Find vector fields for the image and each candidate label.
            # Keep the smallest vector field for each image.
            norm_min = np.inf

            # iterate over all candidate classes
            for target_no in range(1, num_classes):

                target_labels = candidates[0, target_no]
                logits_for_grad = np.zeros_like(logits)
                logits_for_grad[target_labels] = 1
                # gradient of the target label w.r.t. image
                grad_target = _deprocess_gradient(
                    a.backward(logits_for_grad, perturbed),
                    preprocessing, color_axis)

                # Derivative of the binary classifier 'F_lab - F_orig'
                dfx = grad_target - grad_original

                f_im = fx[0, candidates[0, target_no]]

                # create the vector field
                tau_target = _create_tau(
                    f_im, dfx, d1x, d2x, color_axis, smooth
                )

                tau_target += tau_full

                # l2 norm of vector field.
                norm_target = np.linalg.norm(tau_target.ravel())

                # choose the vector field with the smallest norm
                if norm_target < norm_min:
                    norm_min = norm_target
                    tau_min = tau_target

            # Update the image by applying the vector field,
            # the vector field is always applied to the original image,
            # since the current vector field is added to all prior
            # vector fields via tau_target += tau_full
            perturbed_processed = _compose(
                _preprocess_image(
                    image_original.copy(),
                    preprocessing,
                    color_axis),
                tau_min,
                color_axis
            )
            perturbed = _deprocess_image(
                perturbed_processed, preprocessing, color_axis
            )
            tau_full = tau_min
            norm_full = norm_min

            # getting the current label after applying the vector field
            fx, _ = a.predictions(perturbed)
            fx = np.expand_dims(fx, axis=0)
            current_label = np.argmax(fx, axis=1)
            fx = (fx.transpose() - fx[0, current_label]).transpose()

            # See if we have been successful.
            if targeting and (current_label in candidates[0, 1:]):
                vprint('Image successfully deformed from %d to %d.' %
                       (original_label, current_label))
                continue
            elif (not targeting) and current_label != original_label:
                vprint('Image successfully deformed from %d to %d.' %
                       (original_label, current_label))
            vprint('Iterations finished: %d' % n)
            vprint('\tCurrent label: ' + str(current_label))
            vprint('\tnorm(tau) = ' + str(norm_full))
            # vprint('\t(' + str(0) + ')\t' + str( fx[ 0, candidates ] ) )
        # Overshooting
        try_overshoot = False
        have_highest_confidence = (fx == 0)
        predicted_labels = np.where(have_highest_confidence)[0]

        if len(predicted_labels) > 1:
            vprint(
                '\tDeformed image lies on the decision boundary of labels: ' +
                str(predicted_labels)
            )
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
            vprint('\tOvershooting: tau -> %.3f*tau' % os)

            tau_full = os * tau_full
            norm_full = os * norm_full
            perturbed_processed = _compose(
                _preprocess_image(
                    image_original.copy(),
                    preprocessing,
                    color_axis),
                tau_full,
                color_axis
            )
            perturbed = _deprocess_image(
                perturbed_processed, preprocessing, color_axis)

        fx, _ = a.predictions(perturbed)
        fx = np.expand_dims(fx, axis=0)
        current_label = np.argmax(fx, axis=1)
        vprint('\t\t%d -> %d' % (original_label, current_label))

        a.predictions(perturbed)

        self.vector_field = tau_full
        return
