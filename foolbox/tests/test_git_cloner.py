from foolbox.zoo import git_cloner
import os
import hashlib
import pytest
from foolbox.zoo.git_cloner import GitCloneError


def test_git_clone():
    # given
    git_uri = "https://github.com/bethgelab/convex_adversarial.git"
    expected_path = _expected_path(git_uri)

    # when
    path = git_cloner.clone(git_uri)

    # then
    assert path == expected_path


def test_wrong_git_uri():
    git_uri = "git@github.com:bethgelab/non-existing-repo.git"
    with pytest.raises(GitCloneError):
        git_cloner.clone(git_uri)


def _expected_path(git_uri):
    home = os.path.expanduser('~')
    m = hashlib.sha256()
    m.update(git_uri.encode())
    hash = m.hexdigest()
    expected_path = os.path.join(home, '.foolbox_zoo', hash)
    return expected_path
