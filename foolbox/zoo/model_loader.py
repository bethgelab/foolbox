from typing import Any, cast, Optional
from types import ModuleType
import sys
import importlib
import abc
from abc import abstractmethod

from ..models import Model


class ModelLoader(abc.ABC):
    @abstractmethod
    def load(
        self, path: str, module_name: str = "foolbox_model", **kwargs: Any
    ) -> Model:
        """Loads a model from a local path, to which a git repository has been previously cloned to.

        Args:
            path: The path to the local repository containing the code.
            module_name: The name of the module to import.
            kwargs: Additional parameters for the loaded model.

        Returns:
            A foolbox-wrapped model.
        """
        ...

    @staticmethod
    def get(key: Optional[str] = None) -> "ModelLoader":
        if key is None:
            return DefaultLoader()
        else:
            raise ValueError(f"No model loader for: {key}")

    @staticmethod
    def _import_module(path: str, module_name: str = "foolbox_model") -> ModuleType:
        sys.path.insert(0, path)
        module = importlib.import_module(module_name)
        print("imported module: {}".format(module))
        return module


class DefaultLoader(ModelLoader):
    def load(
        self, path: str, module_name: str = "foolbox_model", **kwargs: Any
    ) -> Model:
        module = super()._import_module(path, module_name=module_name)
        model = module.create(**kwargs)  # type: ignore
        return cast(Model, model)
