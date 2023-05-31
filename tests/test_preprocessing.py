import pytest
import pandas as pd
import numpy as np
from titanic_ml_model.data.preprocessing import DataPreprocessor
import os

def test_init():
    sample_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    dp = DataPreprocessor(sample_df)
    assert isinstance(dp, DataPreprocessor)

def test_init_null_exception():
    with pytest.raises(ValueError, match="No DataFrame was provided for DataPreprocessor class."):
        DataPreprocessor(None)

def test_init_input_type_exception():
    # Pass a non-DataFrame object to the initiate method
    non_df_input = "I'm not a DataFrame"

    # Assert that the expected TypeError is raised with the correct error message
    with pytest.raises(TypeError, match="Provided object is not a pandas DataFrame."):
        DataPreprocessor(non_df_input)

def test_init_input_empty_exception():
    # Pass a non-DataFrame object to the initiate method
    sample_df = pd.DataFrame()

    # Assert that the expected TypeError is raised with the correct error message
    with pytest.raises(ValueError, match="Empty DataFrame was provided for DataPreprocessor class."):
        DataPreprocessor(sample_df)

def test_prepare_data():
    sample_df = pd.DataFrame({'passengerid': [1, 2, 3], 'col2': [4, 5, 6]})
    dp = DataPreprocessor(sample_df)
    df_prepared = dp.prepare_data()
    assert 'passengerid' not in df_prepared.columns
    assert 'col2' in df_prepared.columns

def test_feature_engineering():
    sample_df = pd.DataFrame({'cabin': ['A23', 'B45', 'C78'], 'ticket': ['PC 17599', 'STON/O2. 3101282', '113803'], 'name': ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Montvila, Rev. Juozas']})
    dp = DataPreprocessor(sample_df)
    df_engineered = dp.feature_engineering()
    assert 'cabin_number' in df_engineered.columns
    assert 'cabin_letter' in df_engineered.columns
    assert 'cabin_size' in df_engineered.columns
    assert 'ticket_label' in df_engineered.columns
    assert 'ticket_number' in df_engineered.columns
    assert 'name_title' in df_engineered.columns

def test_handle_missing_values():
    sample_df = pd.DataFrame({'cabin': ['A23', 'B45', 'C78'], 'ticket': ['PC 17599', 'STON/O2. 3101282', '113803'], 'name': ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Montvila, Rev. Juozas']})
    dp = DataPreprocessor(sample_df)
    df_engineered = dp.feature_engineering()
    df_handled = dp.handle_missing_values(df_engineered)
    assert not df_handled.isnull().any().any()

def test_transform_data():
    sample_df = pd.DataFrame({
        'cabin_letter': ['A', 'B', 'C'],
        'cabin_number': [121, 313, 131], 
        'age': [31, 21, 21], 
        'ticket_label': ['PC', 'STON/O2.', 'RT'], 
        'name_title': ['Mr', 'Miss', 'Rev'], 
        'sex': ['male', 'female', 'male'],
        'embarked': ['S','C','S'],
        'survived': [1, 0, 1]
    })
    dp = DataPreprocessor(sample_df)
    df_transformed = dp.transform_data()
    assert 'sex' in df_transformed.columns

def test_save_load():
    sample_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    dp = DataPreprocessor(sample_df)
    dp.save('./test.pkl')
    dp_loaded = DataPreprocessor.load('./test.pkl')

    # Assert that the loaded object is an instance of DataPreprocessor
    assert isinstance(dp_loaded, DataPreprocessor)

    # Assert that the loaded object has the same attributes as the original object
    assert dp_loaded.train_df.equals(dp.train_df)
    os.remove('./test.pkl')

def test_save_predictions():
    sample_df = pd.DataFrame({'passengerid': [1, 2, 3]})
    yhat = np.array([0, 1, 1])
    output_path = './predictions.csv'

    # Save predictions
    DataPreprocessor.save_predictions(output_path, sample_df, yhat)

    # Assert that the output file exists
    assert os.path.exists(output_path)

    # Assert that the saved predictions file has the expected content
    saved_df = pd.read_csv(output_path)
    expected_df = pd.DataFrame({'passengerid': [1, 2, 3], 'survived': [0, 1, 1]})
    assert saved_df.equals(expected_df)
    os.remove('./predictions.csv')
