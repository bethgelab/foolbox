from typing import Any
import sys
import pytest

import foolbox as fbn
from foolbox.zoo.model_loader import ModelLoader


@pytest.fixture(autouse=True)
def unload_foolbox_model_module() -> None:
    # reload foolbox_model from scratch for every run
    # to ensure atomic tests without side effects
    module_names = ["foolbox_model", "model"]
    for module_name in module_names:
        if module_name in sys.modules:
            del sys.modules[module_name]


# test_data = [
#     # private repo won't work on travis
#     ("https://github.com/bethgelab/AnalysisBySynthesis.git", (1, 28, 28)),
#     ("https://github.com/bethgelab/convex_adversarial.git", (1, 28, 28)),
#     ("https://github.com/bethgelab/mnist_challenge.git", 784),
#     (join("file://", dirname(__file__), "data/model_repo"), (3, 224, 224)),
# ]

urls = [
    "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications",
    "git@github.com:jonasrauber/foolbox-tensorflow-keras-applications.git",
]


# @pytest.mark.parametrize("url, dim", test_data)
@pytest.mark.parametrize("url", urls)
def test_loading_model(request: Any, url: str) -> None:
    backend = request.config.option.backend
    if backend != "tensorflow":
        pytest.skip()

    # download model
    try:
        fmodel = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)
    except fbn.zoo.GitCloneError:
        pytest.skip()

    # download again (test overwriting)
    try:
        fmodel = fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)
    except fbn.zoo.GitCloneError:
        pytest.skip()

    # create a dummy image
    # x = np.zeros(dim, dtype=np.float32)
    # x[:] = np.random.randn(*x.shape)
    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)

    # run the model
    # logits = model(x)
    # probabilities = ep.softmax(logits)
    # predicted_class = np.argmax(logits)
    assert fbn.accuracy(fmodel, x, y) > 0.9

    # sanity check
    # assert predicted_class >= 0
    # assert np.sum(probabilities) >= 0.9999

    # TODO: delete fmodel


# @pytest.mark.parametrize("url, dim", test_data)
def test_loading_invalid_model(request: Any) -> None:
    url = "https://github.com/jonasrauber/invalid-url"
    with pytest.raises(fbn.zoo.GitCloneError):
        fbn.zoo.get_model(url, name="MobileNetV2", overwrite=True)


def test_non_default_module_throws_error() -> None:
    with pytest.raises(ValueError):
        ModelLoader.get(key="other")
