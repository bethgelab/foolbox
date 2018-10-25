import hashlib
import os
from pathlib import Path


def sha256_hash(git_uri: str):
    m = hashlib.sha256()
    m.update(git_uri.encode())
    return m.hexdigest()


def home_directory_path(folder: str, hash_digest: str):
    # does this work on all operating systems?
    home = str(Path.home())
    return os.path.join(home, folder, hash_digest)


def path_exists(local_path: str):
    return os.path.exists(local_path)
