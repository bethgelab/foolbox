from typing import Any

from ..models import Model

from .git_cloner import clone
from .model_loader import ModelLoader


def get_model(
    url: str, module_name: str = "foolbox_model", overwrite: bool = False, **kwargs: Any
) -> Model:
    """Download a Foolbox-compatible model from the given Git repository URL.

    Examples
    --------

    Instantiate a model:

    >>> from foolbox import zoo
    >>> url = "https://github.com/bveliqi/foolbox-zoo-dummy.git"
    >>> model = zoo.get_model(url)  # doctest: +SKIP

    Only works with a foolbox-zoo compatible repository.
    I.e. models need to have a `foolbox_model.py` file
    with a `create()`-function, which returns a foolbox-wrapped model.

    Using the kwargs parameter it is possible to input an arbitrary number
    of parameters to this methods call. These parameters are forwarded to
    the instantiated model.

    Example repositories:

        - https://github.com/jonasrauber/foolbox-tensorflow-keras-applications
        - https://github.com/bethgelab/AnalysisBySynthesis
        - https://github.com/bethgelab/mnist_challenge
        - https://github.com/bethgelab/cifar10_challenge
        - https://github.com/bethgelab/convex_adversarial
        - https://github.com/wielandbrendel/logit-pairing-foolbox.git
        - https://github.com/bethgelab/defensive-distillation.git

    Args:
        url: URL to the git repository.
        module_name: The name of the module to import.
        kwargs: Optional set of parameters that will be used by the to be instantiated model.

    Returns:
        A Foolbox-wrapped model instance.
    """
    repo_path = clone(url, overwrite=overwrite)
    loader = ModelLoader.get()
    model = loader.load(repo_path, module_name=module_name, **kwargs)
    return model
