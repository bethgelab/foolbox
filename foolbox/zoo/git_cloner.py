import os
import shutil
from git import Repo
import logging
from .common import sha256_hash, home_directory_path

FOLDER = ".foolbox_zoo"


class GitCloneError(RuntimeError):
    pass


def clone(git_uri: str, overwrite: bool = False) -> str:
    """Clones a remote git repository to a local path.

    Args:
        git_uri: The URI to the git repository to be cloned.
        overwrite: Whether or not to overwrite the local path.

    Returns:
        The generated local path where the repository has been cloned to.
    """
    hash_digest = sha256_hash(git_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    exists_locally = os.path.exists(local_path)

    if exists_locally and overwrite:
        # TODO: ideally we would just pull the latest changes instead of cloning again
        shutil.rmtree(local_path, ignore_errors=True)
        exists_locally = False

    if not exists_locally:
        _clone_repo(git_uri, local_path)
    else:
        logging.info(  # pragma: no cover
            "Git repository already exists locally."
        )  # pragma: no cover

    return local_path


def _clone_repo(git_uri: str, local_path: str) -> None:
    logging.info("Cloning repo %s to %s", git_uri, local_path)
    try:
        Repo.clone_from(git_uri, local_path)
    except Exception as e:
        logging.exception("Failed to clone repository", e)
        raise GitCloneError("Failed to clone repository")
    logging.info("Cloned repo successfully.")
