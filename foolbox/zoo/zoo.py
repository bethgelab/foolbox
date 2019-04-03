from .git_cloner import clone
from .model_loader import ModelLoader


def get_model(url, module_name='foolbox_model', **kwargs):
    """

    Provides utilities to download foolbox-compatible robust models
    to easily test attacks against them by simply providing a git-URL.

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

        - https://github.com/bethgelab/AnalysisBySynthesis
        - https://github.com/bethgelab/mnist_challenge
        - https://github.com/bethgelab/cifar10_challenge
        - https://github.com/bethgelab/convex_adversarial
        - https://github.com/wielandbrendel/logit-pairing-foolbox.git
        - https://github.com/bethgelab/defensive-distillation.git

    :param url: URL to the git repository
    :param module_name: the name of the module to import
    :param kwargs: Optional set of parameters that will be used by the
        to be instantiated model.
    :return: a foolbox-wrapped model instance
    """
    repo_path = clone(url)
    loader = ModelLoader.get()
    model = loader.load(repo_path, module_name=module_name, **kwargs)
    return model
