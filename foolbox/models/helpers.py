from __future__ import print_function

import subprocess
from subprocess import PIPE
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
        The model returned by the create() function.
    """

    if path is None:
        path = url.split('/')
        path = path[-1] if path[-1] != '' else path[-2]
        path = path[:-4] if path.endswith('.git') else path
        path = './' + path

    if not os.path.exists(join(path, 'foolbox_model.py')):
        cmd = ['git', 'clone', url, path]
        print('running "{}"'.format(' '.join(cmd)))
        p = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
        output = p.communicate()
        if p.returncode != 0:
            print('"{}" failed with exit code {}'.format(
                ' '.join(cmd), p.returncode))
            stdout = output[0].decode("utf-8")
            stderr = output[1].decode("utf-8")
            if len(stdout) > 0:
                print(stdout)
            if len(stderr) > 0:
                print(stderr)
            return

    if not os.path.exists(join(path, 'foolbox_model.py')):
        raise ValueError('The repository does not contain a '
                         'foolbox_model.py file')

    sys.path.insert(0, path)
    module = importlib.import_module('foolbox_model')
    print('imported module: {}'.format(module))
    return module.create()
