import subprocess
import os
import sys
from os.path import join
import importlib


def from_git(url, path=None):
    """Clones a git repository and loads the model
    from the foolbox_model.py file in the repository.

    Parameters
    ----------
    url : str
        The URL to the git repository.
    path : str
        The name of new directory to clone into.

    Returns
    -------
    `foolbox.model.Model`
        The model returned by the create_model
        function.
    """

    if path is None:
        path = path.split('/')
        path = path[-1] if path[-1] != '' else path[-2]
        path = path[:-4] if path.endswith('.git') else path
        path = './' + path

    try:
        subprocess.check_call(['git', 'clone', url, path])
    except subprocess.CalledProcessError as e:
        print('git clone failed: {}'.format(e))

    if not os.path.exists(join(path, 'foolbox_model.py')):
        raise ValueError('The repository does not contain a '
                         'foolbox_model.py file')

    sys.path.insert(0, path)
    module = importlib.import_module('foolbox_model')
    print('imported module: {}'.format(module))
    return module.create_model()
