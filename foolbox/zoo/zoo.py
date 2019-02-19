from .git_cloner import clone
from .model_loader import ModelLoader


def get_model(url, module_name='foolbox_model'):
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

    Using the module_name parameter it can be specified which module
    should be loaded. This is useful when multiple models are stored
    in the same repository.

    Example repositories:

        - https://github.com/bethgelab/AnalysisBySynthesis
        - https://github.com/bethgelab/mnist_challenge
        - https://github.com/bethgelab/cifar10_challenge
        - https://github.com/bethgelab/convex_adversarial
        - https://github.com/wielandbrendel/logit-pairing-foolbox.git
        - https://github.com/bethgelab/defensive-distillation.git

    :param url: URL to the git repository
    :param module_name: Name of the module to be loaded
    :return: a foolbox-wrapped model instance
    """
    repo_path = clone(url)
    loader = ModelLoader.get()
    model = loader.load(repo_path, module_name)

    return model
