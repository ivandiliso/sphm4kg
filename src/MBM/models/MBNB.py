#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:46:43 2022

@author: nico
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from scipy.special import logsumexp
from owlready2 import *




class MBNB(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        unique, counts = np.unique(y,return_counts=True)
        unique = list(unique)

        c0 = counts[unique.index(0)]
        c1 = counts[unique.index(1)]
 
        self.X_ = X
        self.classes_ = [0,1]
        self.n_classes_ = len(self.classes_)

        self.p_ = np.array([self.X_[np.where(y==i)].mean(axis=0) for i in range(self.n_classes_)])

        self.priors_ = [c0 / (c1 + c0), c1 / (c1 + c0)]

        return self
    
    # Extract Deterministic Rule
    ################################################################################

    def extract_rule(self, classes, theta_classes=0.5):
        
        epsilon = 1e-10
        p = np.clip(self.p_, epsilon, 1 - epsilon)

        mask = p[1] > theta_classes
        owl_selected_classes = [cls for cls, isselected in zip(classes, mask) if isselected]
        rule = And(owl_selected_classes)

        return rule
    
    
    # Predict using Probabilistic Model
    ################################################################################

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        epsilon = 1e-10
        p = np.clip(self.p_, epsilon, 1 - epsilon)
        log_likelihoods = X.dot(np.log(p).T) + (1 - X).dot(np.log(1 - p).T)
        likelihoods = self.priors_ * np.exp(log_likelihoods) 
        likelihoods = likelihoods / likelihoods.sum(axis=1).reshape(-1, 1)

        return likelihoods
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        res = self.predict_proba(X)

        return res.argmax(axis=1)