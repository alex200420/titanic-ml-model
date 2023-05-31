# Titanic ML Model

This package provides a simple machine learning model to predict survival on the Titanic, based on the famous [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). 

## Requirements

- Python 3.6 or later

## Installation

To install the package, clone the repository and install the package with pip:

```bash
git clone https://github.com/username/titanic-ml-model.git
cd titanic-ml-model
pip install .
```
## Usage
After installation, you can use the command-line utilities **`titanic_ml_model train`** and **`titanic_ml_model predict`** to train a model and make predictions.

## Training a model
To train a model, you need to provide a path to the training data. The training data should be in CSV format and match the structure of the Titanic data set on Kaggle.

Here is an example command:

```bash
python -m titanic_ml_model train --train_data ./data/raw/train.csv
```
OR
```bash
titanic_ml_model train --train_data ./data/raw/train.csv
```
OR (For Local Executions: No PIP Installation Required)
```bash
python titanic-runner.py train --train_data ./data/raw/train.csv
```

This command will train a model using the data in **`./data/raw/train.csv`** and save the trained model and the data preprocessor to **`titanic_model.pkl`** and **`titanic_preprocessor.pkl`** respectively.

## Making predictions
After you've trained a model, you can use it to make predictions. You need to provide a path to the data you want to predict.

Here is an example command:

```bash
python -m titanic_ml_model predict --predict_data ./data/raw/test.csv --out_path ./data/out/test.csv
```
OR
```bash
titanic_ml_model predict --predict_data ./data/raw/test.csv --out_path ./data/out/test.csv
```
OR (For Local Executions: No PIP Installation Required)
```bash
python titanic-runner.py predict --predict_data ./data/raw/test.csv --out_path ./data/out/test.csv
```

This command will load the model and preprocessor saved during training, predict survival for the passengers in **`./data/raw/test.csv`**, and write the predictions to **`./data/out/test.csv`**.

# Building Documentation

Sphinx Documentation can be executed by running the following:

```bash
pip install sphinx
cd docs
sphinx-apidoc -o . ../titanic_ml_model
make html
```