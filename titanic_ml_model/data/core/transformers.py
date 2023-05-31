# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import List

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for encoding categorical features using mean of target variable.

    Attributes
    ----------
    cols : List[str]
        List of columns to be encoded
    tgt_col : str
        The target column for mean encoding
    target_encoder : dict
        Stores the mean encoding for each column
    """
    def __init__(self, cols: List[str], tgt_col: str):
        """
        Initialize the TargetEncoderTransformer with column names for encoding and the target column.

        Parameters
        ----------
        cols : List[str]
            List of columns to be encoded
        tgt_col : str
            The target column for mean encoding
        """
        self.cols = cols
        self.tgt_col = tgt_col
        self.target_encoder = {}
        return None

    def fit(self, X):
        """
        Learn encoding from the provided data.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame
        """
        X = X.copy()
        for col in self.cols:
            self.target_encoder[col] = X.groupby(col)[self.tgt_col].mean()
        return self

    def transform(self, X):
        """
        Apply encoding learned during fit.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame
        """
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.target_encoder[col])
            X[col] = X[col].fillna(X[col].mean()) # Fill any remaining NaNs with means
        return X
        
class FillMeanTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for filling missing values with the mean.

    Attributes
    ----------
    cols : List[str]
        List of columns to be encoded
    mean_encoder : dict
        Stores the mean for each column
    """
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.mean_encoder = {}
        return None

    def fit(self, X):
        """
        Learn encoding from the provided data.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame
        """
        X = X.copy()
        for col in self.cols:
            self.mean_encoder[col] = X[col].mean()
        return self

    def transform(self, X):
        """
        Apply encoding learned during fit.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame
        """
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.mean_encoder[col])
        return X
    
class LabelTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for encoding categorical features using sklearn's LabelEncoder.

    Attributes
    ----------
    cols : List[str]
        List of columns to be encoded
    label_encoder : dict
        Stores the label encoding for each column
    """
    def __init__(self, cols: List[str]):
        """
        Initialize the LabelTransformer with column names for encoding.

        Parameters
        ----------
        cols : List[str]
            List of columns to be encoded
        """
        self.cols = cols
        self.label_encoder = {}
        return None

    def fit(self, X):
        """
        Learn encoding from the provided data.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame
        """
        X = X.copy()
        for col in self.cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoder[col] = le
        return self

    def transform(self, X):
        """
        Apply encoding learned during fit.

        Parameters
        ----------
        X : DataFrame
            Input DataFrame
        """
        X = X.copy()
        for col in self.cols:
            X[col] = self.label_encoder[col].transform(X[col])
        return X