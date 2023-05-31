"""
TitanicML - Model Library to solve Titanic Disaster Problem
https://github.com/alex200420/titanic-ml-model
"""
import os.path
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import codecs

def read_text(filepath):
    with codecs.open(filepath, "r", encoding="utf-8") as f:
        return f.read()

here = os.path.dirname(__file__)
# Get the long description from the README file
long_description = read_text(os.path.join(here, 'README.md'))

def read_version_string(version_file):
    for line in read_text(version_file).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = read_version_string("titanic_ml_model/version.py")

requirements = read_text(os.path.join(here, 'requirements.txt')).splitlines()

setup(
    name='titanic_ml_model',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'titanic_ml_model=titanic_ml_model.cli:main'
        ],
    },
    author='Alejandro Jimenez',
    author_email='alejandrob.jimenezp@gmail.com',
    description='A machine learning model for predicting Titanic survivors',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/alex200420/titanic-ml-model',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
    ],
)
