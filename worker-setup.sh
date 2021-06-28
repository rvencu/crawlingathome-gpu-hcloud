#!/bin/bash

git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r worker-requirements.txt --no-cache-dir

yes | pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
