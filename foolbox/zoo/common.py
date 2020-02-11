import hashlib
import os


def sha256_hash(git_uri: str) -> str:
    m = hashlib.sha256()
    m.update(git_uri.encode())
    return m.hexdigest()


def home_directory_path(folder: str, hash_digest: str) -> str:
    # does this work on all operating systems?
    home = os.path.expanduser("~")
    return os.path.join(home, folder, hash_digest)
