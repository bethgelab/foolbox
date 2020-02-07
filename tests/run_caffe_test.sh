#!/bin/bash
# Only python2 caffe image is provided now.
# See https://github.com/BVLC/caffe/issues/5781

pip install -r requirements-dev.txt
pip install mock
pip install --upgrade "numpy<1.16"
# pip install --upgrade scikit-image
pip install -e .
mkdir /mock
cat >>/mock/tensorflow.py<<EOF
__version__ = "mock"
EOF
cat >>/mock/torch.py<<EOF
__version__ = "mock"
EOF

# clear cache for importing mock modules without conflicts
find . -type d -name __pycache__  -o \( -type f -name '*.py[co]' \) -print | xargs rm -rf

PYTHONPATH="/mock:${PYTHONPATH}" pytest --cov-append foolbox/tests/models/test_models_caffe.py
