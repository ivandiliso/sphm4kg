"""
University of Bari Aldo Moro

@author: Nicola Fanizzi, Ivan Diliso
"""

import numpy as np
from owlready2 import *
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MBNB(BaseEstimator, ClassifierMixin):

    def fit(self, X: np.array, y: np.array):
        """Train the MBNB model

        Args:
            X (np.array): Input training data (NxK)
            y (np.array): Ground truth (Nx1)
        """

        X, y = check_X_y(X, y)
        unique, counts = np.unique(y, return_counts=True)
        unique = list(unique)

        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]

        self.X_ = X
        self.classes_ = [0, 1]
        self.n_classes_ = len(self.classes_)

        self.p_ = np.array(
            [self.X_[np.where(y == i)].mean(axis=0) for i in range(self.n_classes_)]
        )

        self.priors_ = [c0 / (c1 + c0), c1 / (c1 + c0)]

        return self

    def extract_rule(self, classes: list, theta_classes: float = 0.5) -> And:
        """Extrac the conjunctive deterministc axiom from the model

        Args:
            classes (list): List of feature names (as owlready2 class objects)
            theta_classes (float, optional): Threshold for inclusion of a feature in the axiom. Defaults to 0.5.

        Returns:
            And: Conjunctive axiom
        """

        epsilon = 1e-10
        p = np.clip(self.p_, epsilon, 1 - epsilon)

        mask = p[1] > theta_classes
        owl_selected_classes = [
            cls for cls, isselected in zip(classes, mask) if isselected
        ]
        rule = And(owl_selected_classes)

        return rule


    def predict_proba(self, X: np.array) -> np.array:
        """Prediction probabilities on the two classes

        Args:
            X (np.array): Input data (NxK)

        Returns:
            np.array: Probability on the binary classes (Nx2)
        """
        check_is_fitted(self)
        X = check_array(X)

        epsilon = 1e-10
        p = np.clip(self.p_, epsilon, 1 - epsilon)
        log_likelihoods = X.dot(np.log(p).T) + (1 - X).dot(np.log(1 - p).T)
        likelihoods = self.priors_ * np.exp(log_likelihoods)
        likelihoods = likelihoods / likelihoods.sum(axis=1).reshape(-1, 1)

        return likelihoods

    def predict(self, X: np.array) -> np.array:
        """Prediction on the true class

        Args:
            X (np.array): Input data (NxK)

        Returns:
            np.array: Predicted class (Nx1)
        """
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)
