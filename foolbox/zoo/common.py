import hashlib
import os


def sha256_hash(git_uri):
    m = hashlib.sha256()
    m.update(git_uri.encode())
    return m.hexdigest()


def home_directory_path(folder, hash_digest):
    # does this work on all operating systems?
    home = os.path.expanduser('~')
    return os.path.join(home, folder, hash_digest)


def path_exists(local_path):
    return os.path.exists(local_path)
