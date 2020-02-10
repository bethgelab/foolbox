import requests
import shutil
import zipfile
import tarfile
import os
import logging

from .common import sha256_hash, home_directory_path, path_exists

FOLDER = ".foolbox_zoo/weights"


def fetch_weights(weights_uri, unzip=False):
    """

    Provides utilities to download and extract packages
    containing model weights when creating foolbox-zoo compatible
    repositories, if the weights are not part of the repository itself.

    Examples
    --------

    Download and unzip weights:

    >>> from foolbox import zoo
    >>> url = 'https://github.com/MadryLab/mnist_challenge_models/raw/master/secret.zip'  # noqa F501
    >>> weights_path = zoo.fetch_weights(url, unzip=True)

    :param weights_uri: the URI to fetch the weights from
    :param unzip: should be `True` if the file to be downloaded is
           a zipped package
    :return: local path where the weights have been downloaded
             and potentially unzipped to
    """
    if weights_uri is None:
        logging.info("No weights to be fetched for this model.")
        return

    hash_digest = sha256_hash(weights_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    exists_locally = path_exists(local_path)

    filename = _filename_from_uri(weights_uri)
    file_path = os.path.join(local_path, filename)

    if exists_locally:
        logging.info("Weights already stored locally.")  # pragma: no cover
    else:
        _download(file_path, weights_uri, local_path)

    if unzip:
        file_path = _extract(local_path, filename)

    return file_path


def _filename_from_uri(url):
    # get last part of the URI, i.e. file-name
    filename = url.split("/")[-1]
    # remove query params if exist
    filename = filename.split("?")[0]
    return filename


def _download(file_path, url, directory):
    logging.info("Downloading weights: %s to %s", url, file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # first check ETag or If-Modified-Since header or similar
    # to check whether updated weights are available?
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise RuntimeError("Failed to fetch weights from %s", url)


def _extract(directory, filename):
    file_path = os.path.join(directory, filename)
    extracted_folder = filename.rsplit(".", 1)[0]
    extracted_folder = os.path.join(directory, extracted_folder)

    if not os.path.exists(extracted_folder):
        logging.info("Extracting weights package to %s", extracted_folder)
        os.makedirs(extracted_folder)
        if ".zip" in file_path:
            zip_ref = zipfile.ZipFile(file_path, "r")
            zip_ref.extractall(extracted_folder)
            zip_ref.close()
        elif ".tar.gz" in file_path:  # pragma: no cover
            tar_ref = tarfile.TarFile.open(file_path, "r")
            tar_ref.extractall(extracted_folder)
            tar_ref.close()
    else:
        logging.info(
            "Extraced folder already exists: %s", extracted_folder
        )  # pragma: no cover

    return extracted_folder
