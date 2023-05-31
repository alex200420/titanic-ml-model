import pytest
import subprocess
import pandas as pd
import os
from pathlib import Path
import numpy as np
from titanic_ml_model.titanic_cli import TitanicCLI
from titanic_ml_model.models.model import TitanicModel
from titanic_ml_model.data.preprocessing import DataPreprocessor


# Define the data for the default train dataframe
default_train_data = {
    "PassengerId": np.arange(1, 11),
    "Survived": np.random.randint(0, 2, 10),
    "Pclass": np.random.randint(1, 4, 10),
    "Name": ["Braund, Mr. Owen Harris"]*10,
    "Sex": np.random.choice(["male", "female"], 10),
    "Age": np.random.uniform(20, 50, 10),
    "SibSp": np.random.randint(0, 2, 10),
    "Parch": np.random.randint(0, 2, 10),
    "Ticket": ["Ticket" + str(i) for i in range(1, 11)],
    "Fare": np.random.uniform(10, 50, 10),
    "Cabin": ["Cabin" + str(i) for i in range(1, 11)],
    "Embarked": np.random.choice(["S", "C", "Q"], 10)
}

# Define the data for the default test dataframe
default_test_data = {
    "PassengerId": np.arange(1, 11),
    "Pclass": np.random.randint(1, 4, 10),
    "Name": ["Braund, Mr. Owen Harris"]*10,
    "Sex": np.random.choice(["male", "female"], 10),
    "Age": np.random.uniform(20, 50, 10),
    "SibSp": np.random.randint(0, 2, 10),
    "Parch": np.random.randint(0, 2, 10),
    "Ticket": ["Ticket" + str(i) for i in range(1, 11)],
    "Fare": np.random.uniform(10, 50, 10),
    "Cabin": ["Cabin" + str(i) for i in range(1, 11)],
    "Embarked": np.random.choice(["S", "C", "Q"], 10)
}

# Create default dataframes
default_train_df = pd.DataFrame(default_train_data)
default_test_df = pd.DataFrame(default_test_data)

# This is the function that will replace pd.read_csv
def mock_read_csv(file_path, *args, **kwargs):
    if file_path == 'train.csv':
        return default_train_df.copy()
    elif file_path == 'test.csv':
        return default_test_df.copy()

@pytest.fixture(autouse=True)
def mock_default_df(monkeypatch):
    """Replace pd.read_csv with a function that returns a default dataframe"""
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

def test_initialization():
    cli = TitanicCLI('model.pkl', 'preprocessor.pkl')
    assert cli.model_path == 'model.pkl'
    assert cli.preprocessor_path == 'preprocessor.pkl'

def test_train():
    cli = TitanicCLI('./model.pkl', './preprocessor.pkl')
    cli.train('train.csv')

    # Check that model and preprocessor files were created
    assert os.path.exists('./model.pkl')
    assert os.path.exists('./preprocessor.pkl')

    # Load the saved model and preprocessor, check they have expected attributes
    model = TitanicModel.load('./model.pkl')
    preprocessor = DataPreprocessor.load('./preprocessor.pkl')
    assert hasattr(model, 'model')
    assert hasattr(model, 'scaler')
    assert hasattr(preprocessor, 'train_df')
    assert hasattr(preprocessor, 'train_y')

def test_predict():
    cli = TitanicCLI('./model.pkl', './preprocessor.pkl')
    cli.predict('test.csv', './predictions.csv')

    # Check that predictions file was created
    assert os.path.exists('./predictions.csv')
    os.remove('./model.pkl')
    os.remove('./preprocessor.pkl')
    os.remove('./predictions.csv')