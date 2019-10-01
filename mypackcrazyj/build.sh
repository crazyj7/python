#!/bin/sh
rm -rf ./build
rm -rf ./dist
python setup.py sdist bdist_wheel
python -m twine upload -u crazyj --repository-url https://test.pypi.org/legacy/ dist/* --verbose
