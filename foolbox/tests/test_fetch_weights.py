from foolbox.zoo import fetch_weights
from foolbox.zoo.common import path_exists, home_directory_path, sha256_hash
from foolbox.zoo.weights_fetcher import FOLDER

import os
import pytest
import shutil

import responses
import io
import zipfile


@responses.activate
def test_fetch_weights_unzipped():
    weights_uri = 'http://localhost:8080/weights.zip'
    raw_body = _random_body(zipped=False)

    # mock server
    responses.add(responses.GET, weights_uri,
                  body=raw_body, status=200, stream=True)

    expected_path = _expected_path(weights_uri)

    if path_exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    file_path = fetch_weights(weights_uri)

    exists_locally = path_exists(expected_path)
    assert exists_locally
    assert expected_path in file_path


@responses.activate
def test_fetch_weights_zipped():
    weights_uri = 'http://localhost:8080/weights.zip'

    # mock server
    raw_body = _random_body(zipped=True)
    responses.add(responses.GET, weights_uri,
                  body=raw_body, status=200, stream=True,
                  content_type='application/zip',
                  headers={'Accept-Encoding': 'gzip, deflate'})

    expected_path = _expected_path(weights_uri)

    if path_exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    file_path = fetch_weights(weights_uri, unzip=True)

    exists_locally = path_exists(expected_path)
    assert exists_locally
    assert expected_path in file_path


@responses.activate
def test_fetch_weights_returns_404():
    weights_uri = 'http://down:8080/weights.zip'

    # mock server
    responses.add(responses.GET, weights_uri, status=404)

    expected_path = _expected_path(weights_uri)

    if path_exists(expected_path):
        shutil.rmtree(expected_path)  # make sure path does not exist already

    with pytest.raises(RuntimeError):
        fetch_weights(weights_uri, unzip=False)


def test_no_uri_given():
    assert fetch_weights(None) is None


def _random_body(zipped=False):
    if zipped:
        data = io.BytesIO()
        with zipfile.ZipFile(data, mode='w') as z:
            z.writestr('test.txt', 'no real weights in here :)')
        data.seek(0)
        return data.getvalue()
    else:
        raw_body = os.urandom(1024)
        return raw_body


def _expected_path(weights_uri):
    hash_digest = sha256_hash(weights_uri)
    local_path = home_directory_path(FOLDER, hash_digest)
    return local_path
