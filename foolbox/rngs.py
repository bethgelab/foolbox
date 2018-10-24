import random
import numpy as np

rng = random.Random()
nprng = np.random.RandomState()


def set_seeds(seed):
    """Sets the seeds of both random number generators used by Foolbox.

    Parameters
    ----------
    seed : int
        The seed for both random number generators.

    """
    rng.seed(seed)
    nprng.seed(seed)
