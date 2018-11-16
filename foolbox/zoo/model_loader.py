import sys
import importlib

import abc
abstractmethod = abc.abstractmethod
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})


class ModelLoader(ABC):

    @abstractmethod
    def load(self, path):
        """
        Load a model from a local path, to which a git repository
        has been previously cloned to.

        :param path: the path to the local repository containing the code
        :return: a foolbox-wrapped model
        """
        pass  # pragma: no cover

    @staticmethod
    def get(key='default'):
        if key is 'default':
            return DefaultLoader()
        else:
            raise RuntimeError("No model loader for: %s".format(key))

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
