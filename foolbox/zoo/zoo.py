from .git_cloner import clone
from .model_loader import ModelLoader


def get_model(url):
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

    Example repositories:

        - https://github.com/bethgelab/mnist_challenge
        - https://github.com/bethgelab/cifar10_challenge
        - https://github.com/bethgelab/convex_adversarial

    :param url: URL to the git repository
    :return: a foolbox-wrapped model instance
    """
    repo_path = clone(url)
    loader = ModelLoader.get()
    model = loader.load(repo_path)

    return model
