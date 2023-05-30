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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import shap
import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TitanicModel:
    def __init__(self, n_iter_search: int = 20, cv_splits = 5):
        self.model = HistGradientBoostingClassifier()
        self.param_dist = {'max_depth': [3, 5, 7, 10, None],
                           'min_samples_leaf': [1, 2, 5, 10],
                           'learning_rate': np.logspace(-3, 0, 10),
                           'l2_regularization': [0.0, 0.1, 1.0]}
        self.n_iter_search = n_iter_search
        self.cv_splits = cv_splits
        self.sample_X = None
        self.prob_treshold = None
        self.scaler = StandardScaler()

    def train(self, X, y):
        random_search = RandomizedSearchCV(self.model, param_distributions=self.param_dist,
                                           n_iter=self.n_iter_search, cv=StratifiedKFold(n_splits=self.cv_splits), scoring='accuracy', random_state=42)

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        random_search.fit(X_scaled, y)
        self.model = random_search.best_estimator_
        self.sample_X = X.copy()
        self.prob_treshold, _ = self.find_best_threshold(self.model, X_scaled, y)

    def predict(self, X):
        if self.model is None:
            raise Exception("The model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:,1]
        predictions = (probs > self.prob_treshold).astype(int)
        return predictions

    def summarize_model(self):
        if self.model is None:
            raise Exception("The model must be trained before summarizing it")
        
        summary_string = ""
        summary_string += f"Model parameters:\n"
        summary_string += f"Max depth: {self.model.max_depth}\n"
        summary_string += f"Min samples leaf: {self.model.min_samples_leaf}\n"
        summary_string += f"Learning rate: {self.model.learning_rate}\n"
        summary_string += f"L2 regularization: {self.model.l2_regularization}\n"

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.sample_X)

        summary_string += "SHAP values:\n"
        for i in range(len(self.sample_X.columns)):
            summary_string += f"{self.sample_X.columns[i]}: {np.mean(np.abs(shap_values[:, i]))}\n"

        # Generate SHAP summary plot and save it as a png file
        shap.summary_plot(shap_values, self.sample_X, feature_names=self.sample_X.columns, show=False)
        plt.savefig("shap_summary_plot.png")
        logger.info("SHAP summary plot saved as shap_summary_plot.png")

        return summary_string

    def evaluate(self, X, y):
        if self.model is None:
            raise Exception("The model must be trained before evaluating it")

        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        cm = confusion_matrix(y, predictions)

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

        return metrics

    def save(self, path: str) -> None:
        logger.info('Saving TitanicModel')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        logger.info('TitanicModel Succesfully Saved')

    @staticmethod
    def load(path: str) -> 'TitanicModel':
        logger.info('Loading TitanicModel')
        with open(path, 'rb') as file:
            model = pickle.load(file)
            logger.info('TitanicModel Succesfully Loaded')
            return model

    @staticmethod
    def find_best_threshold(model, X, y):
        y_scores = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_scores)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        roc_auc = auc(fpr, tpr)

        return optimal_threshold, roc_auc
