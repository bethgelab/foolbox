import eagerpy as ep

from .devutils import flatten
from .devutils import wrap


# all distances should
# * accept native tensors and EagerPy tensors -> use wrap
# * return the same format as the input -> use restore
# * expect a batch dimension and arbitrary other dimensions -> use flatten
# * accept two arguments, reference and perturbed


def norm_to_distance(norm):
    def distance(reference, perturbed):
        """Calculates the distance from reference to perturbed using the {norm.__name__} norm.

        Parameters
        ----------
        reference : T
            A batch of reference inputs.
        perturbed : T
            A batch of perturbed inputs.

        Returns
        -------
        T
            Returns a 1D tensor with the {norm.__name__} distance for each sample in the batch

        """
        reference, perturbed, restore = wrap(reference, perturbed)
        norms = norm(flatten(perturbed - reference), axis=-1)
        return restore(norms)

    distance.__name__ = norm.__name__
    distance.__qualname__ = norm.__qualname__
    distance.__doc__ = distance.__doc__.format(norm=norm)
    return distance


l0 = norm_to_distance(ep.norms.l0)
l1 = norm_to_distance(ep.norms.l1)
l2 = norm_to_distance(ep.norms.l2)
linf = norm_to_distance(ep.norms.linf)
