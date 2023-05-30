# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class TargetEncoder(BaseEstimator, TransformerMixin): 
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
        
class MeanEncoder(BaseEstimator, TransformerMixin): 
    def __init__(self, cols: List[str], tgt_col: str):
        self.cols = cols
        self.tgt_col = tgt_col
        self.mean_encoder = {}
        return None

    def fit(self, X):
        X = X.copy()
        for col in self.cols:
            self.mean_encoder[col] = X[self.tgt_col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].fillna(self.mean_encoder[col])
        return X