from setuptools import setup

'''
LOCAL: 
install:  pip install .

---------------------------------------
REMOTE:
pip install --user --upgrade twine

## upload
python setup.py sdist bdist_wheel
# test
python -m twine upload -u crazyj --repository-url https://test.pypi.org/legacy/ dist/* --verbose

## install
pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps mypackcrazyj
pip uninstall mpackcrazyj

## RELEASE
# release
python -m twine upload -u crazyj dist/* --verbose
pip install mypackcrazyj
 

'''

setup(name='mypackcrazyj',
      version='0.1.1',
      description='package testing',
      url='http://github.com/crazyj7/python/mypackcrazyj',
      author='crazyj',
      author_email='crazyj7@gmail.com',
      license='MIT',
      packages=['mypackcrazyj'],
      install_requires=['markdown',],
      zip_safe=False,
      python_requires='>=3.6')


'''

install_requires : import files...
python_requires='>=3.6'


'''