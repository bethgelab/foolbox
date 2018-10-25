from abc import ABC, abstractmethod
import sys
import importlib


class ModelLoader(ABC):

    @abstractmethod
    def load(self, path):
        pass  # pragma: no cover

    @staticmethod
    def get(key='default'):
        if key is 'default':
            return DefaultLoader()
        else:
            raise RuntimeError(f"No model loader for: {key}")

    @staticmethod
    def _import_module(path, module_name='foolbox_model'):
        sys.path.insert(0, path)
        module = importlib.import_module(module_name)
        print('imported module: {}'.format(module))
        return module


class DefaultLoader(ModelLoader):

    def load(self, path, module_name='foolbox_model'):
        module = ModelLoader._import_module(path, module_name)
        model = module.create()
        return model
