#!/bin/bash

python -m venv .env
source .env/bin/activate

cd PythonAPI/
python setup.py build_ext --inplace
python setup.py build_ext install
cd ..

python sample.py


pip install opencv-python-headless