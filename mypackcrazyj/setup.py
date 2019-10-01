from setuptools import setup

'''
LOCAL: 
install:  pip install .

---------------------------------------
REMOTE:
pip install --user --upgrade twine

## upload
python setup.py sdist bdist_wheel
# test : https://test.pypi.org/project/mypackcrazyj/
python -m twine upload -u crazyj --repository-url https://test.pypi.org/legacy/ dist/* --verbose

## install
pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps mypackcrazyj
pip uninstall mpackcrazyj

## RELEASE
# release : https://pypi.org/project/mypackcrazyj
python -m twine upload -u crazyj dist/* --verbose
pip install mypackcrazyj
 

'''
import setuptools

with open("README.md", "r") as f:
      long_description = f.read()

setup(
      name='mypackcrazyj',
      version='0.1.4',
      description='package testing',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/crazyj7/python/tree/master/mypackcrazyj',

      author='crazyj',
      author_email='crazyj7@gmail.com',
      license='MIT',

      packages=setuptools.find_packages(),
      install_requires=['markdown',],
      zip_safe=False,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      python_requires = '>=3.6',
)


'''

install_requires : import files...
python_requires='>=3.6'


'''