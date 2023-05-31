from __future__ import annotations
import numpy as np
import pandas as pd
from .core.transformers import TargetEncoderTransformer, FillMeanTransformer, LabelTransformer
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import logging
import pickle
import re
import numpy as np
import os

logging.basicConfig(level=logging.WARNING) #configurated so as to show the expected output in this exercise
logger = logging.getLogger(__name__) # it can be changed to INFO to show all LOGs

class DataPreprocessor:
    """
    A class used to preprocess the Titanic dataset.

    Attributes
    ----------
    train_df : object
        The training DataFrame
    train_y : object
        The training objective
    pipeline : dict
        Transformer Pipeline
    """
    def __init__(self, sample_df):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Parameters
        ----------
            sample_df : pd.DataFrame
                Training DataFrame
        """
        if sample_df is None:
            logger.error("Provided DataFrame is None.")
            raise ValueError("No DataFrame was provided for DataPreprocessor class.")
        
        if not isinstance(sample_df, pd.DataFrame):
            logger.error("Provided sample_df is not a pandas DataFrame.")
            raise TypeError("Provided object is not a pandas DataFrame.")

        if sample_df.empty:
            logger.error("Empty DataFrame was provided for DataPreprocessor class.")
            raise ValueError("Empty DataFrame was provided for DataPreprocessor class.")
        
        self.train_df = sample_df
        self.train_y = None
        self.pipeline = OrderedDict({
            'Target Encoding': TargetEncoderTransformer(['embarked', 'cabin_letter', 'ticket_label', 'name_title'], 'survived'),
            'Fill NA Mean': FillMeanTransformer(['cabin_number', 'age']),
            'Label Encoding': LabelTransformer(['sex'])
        })

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepares the data for feature engineering.

        Parameters
        ----------
        oos_df : pd.DataFrame, optional
            The out-of-sample dataframe. If None, uses the train_df attribute (default is None).

        Returns
        -------
        df : pd.DataFrame
            The prepared dataframe.
        """
        logger.info('Preparing Data ...')
        df = self.train_df if oos_df is None else oos_df
        df.columns = df.columns.str.lower()

        df = df.set_index('passengerid')
        logger.info('Data prepared')
        return df

    def feature_engineering(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Performs feature engineering on the data.

        Parameters
        ----------
        oos_df : pd.DataFrame, optional
            The out-of-sample dataframe. If None, uses the train_df attribute (default is None).

        Returns
        -------
        df : pd.DataFrame
            The dataframe with engineered features.
        """
        logger.info('Engineering Features ...')
        df = self.train_df if oos_df is None else oos_df
        # Method to engineer new features
        df['cabin_number'] = df['cabin'].apply(self._get_cabin_number)
        df['cabin_letter'] = df['cabin'].apply(self._get_cabin_letter)
        df['cabin_size'] = df['cabin'].apply(self._get_cabin_size)
        df['ticket_label'] = df['ticket'].apply(self._get_ticket_label)
        df['ticket_number'] = df['ticket'].apply(self._get_ticket_number)
        df['name_title'] = df['name'].apply(self._get_name_title)
        if oos_df is None:
            self.train_df = df
        logger.info('Features Engineered')
        return df

    def handle_missing_values(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Handles missing values in the data.

        Parameters
        ----------
        oos_df : pd.DataFrame, optional
            The out-of-sample dataframe. If None, uses the train_df attribute (default is None).

        Returns
        -------
        df : pd.DataFrame
            The dataframe with handled missing values.
        """
        logger.info('Filling Missing Values ...')
        df = self.train_df if oos_df is None else oos_df

        # Method to handle missing values
        mask = df['ticket_number'].isnull()
        df.loc[mask, 'ticket_label'] = df.loc[mask, 'ticket']
        df['ticket_label'] = df['ticket_label'].fillna('OTHER')
        df.loc[~df.ticket_label.isin(df.ticket_label.value_counts(1).index[:5].values),'ticket_label'] = 'Misc'
        df['ticket_number'] = np.log1p(df['ticket_number']).fillna(0)
        df['cabin_letter'] = df['cabin_letter'].fillna('U')

        cols = df.columns[df.columns.isin(['pclass','sex','age','sibsp','parch','fare','embarked','cabin_number','cabin_letter','cabin_size',
                     'ticket_label','ticket_number','name_title','survived'])]  
        df = df[cols]
        if oos_df is None:
            self.train_df = df
        logger.info('Missing Values Filled')
        return df
    
    def transform_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Transforms the data using the pipeline.

        Parameters
        ----------
        oos_df : pd.DataFrame, optional
            The out-of-sample dataframe. If None, uses the train_df attribute (default is None).

        Returns
        -------
        df : pd.DataFrame
            The transformed dataframe.
        """
        logger.info('Transforming Data ...')
        df = self.train_df if oos_df is None else oos_df
        for step_name, _ in self.pipeline.items():
            if oos_df is None:
                self.pipeline[step_name].fit(df)
            df = self.pipeline[step_name].transform(df)
        if 'survived' in df.columns:
            self.train_y = df['survived']
            self.train_df = df.drop(['survived'], axis = 1)
        logger.info('Data Transformed')
        return df
    
    def save(self, path: str) -> None:
        """
        Saves the DataPreprocessor to a file.

        Parameters
        ----------
        path : str
            The path where the DataPreprocessor will be saved.
        """
        logger.info('Saving DataPreprocessor')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        logger.info('DataPreprocessor Succesfully Saved')

    @staticmethod
    def load(path: str) -> DataPreprocessor:
        """
        Loads a DataPreprocessor from a file.

        Parameters
        ----------
        path : str
            The path where the DataPreprocessor is located.

        Returns
        -------
        model : DataPreprocessor
            The loaded DataPreprocessor.
        """
        logger.info('Loading DataPreprocessor ...')
        with open(path, 'rb') as file:
            model = pickle.load(file)
            logger.info('DataPreprocessor Succesfully Loaded')
            return model
        
    @staticmethod
    def save_predictions(path:str, df: pd.DataFrame, yhat: np.array) -> None:
        """
        Saves the predictions along with the original data to a CSV file at the given path.

        Parameters
        ----------
        path : str
            Path where the CSV file will be written
        data : pd.DataFrame
            Original data that the predictions are based on
        predictions : np.array
            Predictions to be saved
        """
        logger.info('Saving Predictions ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = df.copy()
        df['survived'] = yhat
        df = df.reset_index()[['passengerid', 'survived']]
        df.to_csv(path, index = False)
        logger.info('Predictions Saved.')

    @staticmethod
    def train_test_split(X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _get_cabin_number(x:str):
        return pd.Series([int(ticket_num) for ticket_num in re.findall("\d+", x)]).mean() if not pd.isnull(x) else np.nan
    
    @staticmethod
    def _get_cabin_letter(x:str):
        return pd.Series.mode(pd.Series([ticket_num for ticket_num in re.findall("[A-Za-z]", x)]))[0] if not pd.isnull(x) else None
    
    @staticmethod
    def _get_cabin_size(x:str):
        return pd.Series([ticket_num for ticket_num in re.findall("[A-Za-z]", x)]).size if not pd.isnull(x) else 0

    @staticmethod
    def _get_ticket_label(x:str):
        return re.findall("(.*) ", x)[0].strip() if len(re.findall("(.*) ", x))>0 else None

    @staticmethod
    def _get_ticket_number(x:str):
        return int(re.findall("^\d+| \d+", x)[0].strip()) if len(re.findall("^\d+| \d+", x)) > 0 else np.nan

    @staticmethod
    def _get_name_title(x:str):
        return x.split(',')[1].split('.')[0].strip()