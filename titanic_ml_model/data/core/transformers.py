# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import List

class TargetEncoderTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self, cols: List[str], tgt_col: str):
        self.cols = cols
        self.tgt_col = tgt_col
        self.target_encoder = {}
        return None

    def fit(self, X):
        X = X.copy()
        for col in self.cols:
            self.target_encoder[col] = X.groupby(col)[self.tgt_col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.target_encoder[col])
        return X
        
class FillMeanTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.mean_encoder = {}
        return None

    def fit(self, X):
        X = X.copy()
        for col in self.cols:
            self.mean_encoder[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.mean_encoder[col])
        return X
    
class LabelTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.label_encoder = {}
        return None

    def fit(self, X):
        X = X.copy()
        for col in self.cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoder[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = self.label_encoder[col].transform(X[col])
        return X