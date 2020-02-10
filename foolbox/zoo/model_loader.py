import sys
import importlib
import abc
from abc import abstractmethod


class ModelLoader(abc.ABC):
    @abstractmethod
    def load(self, path, module_name="foolbox_model", **kwargs):
        """
        Load a model from a local path, to which a git repository
        has been previously cloned to.

        :param path: the path to the local repository containing the code
        :param module_name: the name of the module to import
        :param kwargs: parameters for the to be loaded model
        :return: a foolbox-wrapped model
        """
        pass  # pragma: no cover

    @staticmethod
    def get(key=None):
        if key is None:
            return DefaultLoader()
        else:
            raise RuntimeError("No model loader for: {}".format(key))

    @staticmethod
    def _import_module(path, module_name="foolbox_model"):
        sys.path.insert(0, path)
        module = importlib.import_module(module_name)
        print("imported module: {}".format(module))
        return module


class DefaultLoader(ModelLoader):
    def load(self, path, module_name="foolbox_model", **kwargs):
        module = ModelLoader._import_module(path, module_name=module_name)
        model = module.create(**kwargs)
        return model
