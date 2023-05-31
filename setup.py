from setuptools import setup, find_packages

setup(
    name='titanic_ml_model',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'typeguard',
        'shap',
        'matplotlib'
    ],
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
