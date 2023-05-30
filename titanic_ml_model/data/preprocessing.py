"""
Instructions:

- Fill in the methods of the DataModeler class to produce the same printed results
  as in the comments labeled '<Expected Output>' in the second half of the file.
- The DataModeler should predict the 'outcome' from the columns 'amount' and 'transaction date.'
  Your model should ignore the 'customer_id' column.
- For the modeling methods `fit`, `predict` and `model_summary` you can use any appropriate method.
  Try to get 100% accuracy on both training and test, as indicated in the output.
- Your solution will be judged on both correctness and code quality.
- Good luck, and have fun!

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typeguard import typechecked # this makes sure data types follow Data Type Hints (Optional)
from sklearn.tree import DecisionTreeClassifier
from scipy.special import softmax
from .core.transformers import TargetEncoderTransformer, FillMeanTransformer, LabelTransformer
from collections import OrderedDict
import shap
import logging
import pickle
import re
import numpy as np
from typing import Union

logging.basicConfig(level=logging.WARNING) #configurated so as to show the expected output in this exercise
logger = logging.getLogger(__name__) # it can be changed to INFO to show all LOGs

class DataPreprocessor:
    """
    This class is responsible for the preprocessing of data for the Titanic Machine Learning model.
    """
    def __init__(self, sample_df):
        """
        Initializes a new instance of DataPreprocessor.
        :param sample_df: The sample dataframe to be used for preprocessing.
        """
        if sample_df is None or sample_df.empty:
            logger.error("Provided DataFrame is None or empty.")
            raise ValueError("No DataFrame was provided for DataModeler class.")
        
        if not isinstance(sample_df, pd.DataFrame):
            logger.error("Provided sample_df is not a pandas DataFrame.")
            raise TypeError("Provided object is not a pandas DataFrame.")
        
        self.train_df = sample_df
        self.train_y = None
        self.pipeline = OrderedDict({
            'Target Encoding': TargetEncoderTransformer(['embarked', 'cabin_letter', 'ticket_label', 'name_title'], 'survived'),
            'Fill NA Mean': FillMeanTransformer(['cabin_number', 'age']),
            'Label Encoding': LabelTransformer(['sex'])
        }) 
        self.feature_means = {}

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepares the data for preprocessing.
        :param oos_df: The out-of-sample dataframe. If None, train_df will be used.
        :return: The prepared dataframe.
        """
        logger.info('Preparing Data')
        df = self.train_df if oos_df is None else oos_df
        df.columns = df.columns.str.lower()

        df = df.set_index('passengerid')
        logger.info('Data prepared')
        return df

    def feature_engineering(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineers new features from the existing ones.
        :param oos_df: The out-of-sample dataframe. If None, train_df will be used.
        :return: The dataframe with engineered features.
        """
        logger.info('Engineering Features')
        df = self.train_df if oos_df is None else oos_df
        # Method to engineer new features
        df['cabin_number'] = df['cabin'].apply(self._get_cabin_number)
        df['cabin_letter'] = df['cabin'].apply(self._get_cabin_letter)
        df['cabin_size'] = df['cabin'].apply(self._get_cabin_size)
        df['ticket_label'] = df['ticket'].apply(self._get_ticket_label)
        df['ticket_number'] = df['ticket'].apply(self._get_ticket_number)
        df['name_title'] = df.name.apply(self._get_name_title)
        if oos_df is None:
            self.train_df = df
        logger.info('Features Engineered')
        return df

    def handle_missing_values(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Handles Missing Values in Data
        :param oos_df: The out-of-sample dataframe. If None, train_df will be used.
        :return: The dataframe with no missing features.
        """
        logger.info('Filling Missing Values')
        df = self.train_df if oos_df is None else oos_df

        # Method to handle missing values
        mask = df['ticket_number'].isnull()
        df.loc[mask, 'ticket_label'] = df.loc[mask, 'ticket']
        df['ticket_label'] = df['ticket_label'].fillna('OTHER')
        df.loc[~df.ticket_label.isin(df.ticket_label.value_counts(1).index[:5].values),'ticket_label'] = 'Misc'
        df['ticket_number'] = np.log1p(df['ticket_number']).fillna(0)
        df['cabin_letter'] = df['cabin_letter'].fillna('U')
        df = df[
            ['pclass','sex','age','sibsp','parch','fare','embarked','cabin_number','cabin_letter','cabin_size',
             'ticket_label','ticket_number','name_title','survived'
        ]]
        if oos_df is None:
            self.train_df = df
        logger.info('Missing Values Filled')
        return df
    
    def transform_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Transform Data and Saves Learned Information in Pipeline
        :param oos_df: The out-of-sample dataframe. If None, train_df will be used.
        :return: The dataframe with no missing features.
        """
        logger.info('Transforming Data')
        for step_name, _ in self.pipeline.items():
            if oos_df is None:
                df = self.pipeline[step_name].fit_transform(self.train_df)
                self.train_df = df
            else:
                df = self.pipeline[step_name].transform(oos_df)
        logger.info('Data Transformed')
        return df

    def save(self, path: str) -> None:
        '''
        Save the DataPreprocessor so it can be re-used.
        :param path: Path to save model to.
        '''
        logger.info('Saving DataPreprocessor')
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        logger.info('DataPreprocessor Succesfully Saved')

    @staticmethod
    def load(path: str) -> DataPreprocessor:
        '''
        Load the DataPreprocessor.
        :param path: Path to load model from.
        '''
        logger.info('Loading DataModeler')
        with open(path, 'rb') as file:
            model = pickle.load(file)
            logger.info('DataPreprocessor Succesfully Loaded')
            return model
    
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