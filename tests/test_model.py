import pytest
import pandas as pd
from titanic_ml_model.models.model import TitanicModel
import os 
import numpy as np

# Create a simple dataset for testing
def create_dataset():
    X = pd.DataFrame({
        'Pclass': [3, 1, 3, 4, 2, 1],
        'Age': [22, 38, 26, 31, 23, 41],
        'SibSp': [1, 1, 0, 1, 0, 1],
        'Parch': [0, 0, 0, 0, 0, 0],
        'Fare': [7.25, 71.2833, 71.2833, 71.2833, 7.925, 7.925]
    })
    y = pd.Series([0, 1, 1, 1, 0, 0])

    return X, y


def test_initialization():
    np.random.seed(42)
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    assert model.n_iter_search == 10
    assert model.cv_splits == 2


def test_train():
    np.random.seed(42)
    X, y = create_dataset()
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    model.train(X, y)
    assert model.model is not None


def test_predict():
    np.random.seed(42)
    X, y = create_dataset()
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    model.train(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)

def test_predict_null_exception():
    np.random.seed(42)
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    X, _ = create_dataset()
    # Assert that the expected TypeError is raised with the correct error message
    with pytest.raises(ValueError, match = "The model must be trained before making predictions"):
        model.predict(X)

def test_evaluate():
    np.random.seed(42)
    X, y = create_dataset()
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    model.train(X, y)
    metrics = model.evaluate(X, y)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'confusion_matrix' in metrics

def test_evaluate_null_exception():
    np.random.seed(42)
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    X, y = create_dataset()
    # Assert that the expected TypeError is raised with the correct error message
    with pytest.raises(ValueError, match = "The model must be trained before evaluating it"):
        model.evaluate(X, y)

def test_summarize_model():
    np.random.seed(42)
    X, y = create_dataset()
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    model.train(X, y)
    summary_string = model.summarize_model()
    assert "Model parameters:" in summary_string
    assert "SHAP values:" in summary_string

def test_summarize_model_null_exception():
    np.random.seed(42)
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    # Assert that the expected TypeError is raised with the correct error message
    with pytest.raises(ValueError, match = "The model must be trained before summarizing it"):
        model.summarize_model()

def test_save_load():
    np.random.seed(42)
    X, y = create_dataset()
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    model.train(X, y)
    model.save('./test_model.pkl')
    loaded_model = TitanicModel.load('./test_model.pkl')
    assert loaded_model.model.get_params() == model.model.get_params()
    os.remove('test_model.pkl')


def test_find_best_threshold():
    np.random.seed(42)
    X, y = create_dataset()
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    model.train(X, y)
    print(model.predict(X))
    optimal_threshold, roc_auc = model.find_best_threshold(model.model, X, y)
    #assert 0 <= optimal_threshold <= 1
    assert 0 <= roc_auc <= 1


def test_exceptions():
    np.random.seed(42)
    model = TitanicModel(n_iter_search=10, cv_splits=2)
    X, y = create_dataset()

    with pytest.raises(Exception):
        model.predict(X)

    with pytest.raises(Exception):
        model.summarize_model()

    with pytest.raises(Exception):
        model.evaluate(X, y)
