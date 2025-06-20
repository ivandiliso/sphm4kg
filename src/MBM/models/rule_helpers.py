import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
from abc import ABC, abstractmethod


from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class RulePredictionWrapper(ABC):
    def __init__(self, trained_model, individuals, classes):
        self.trained_model = trained_model
        self.individuals = individuals
        self.classes = classes

    def predict(self, X, **kwargs):
        rule = self.trained_model.extract_rule(self.classes, **kwargs)
        rule_individuals = self._compute_rule(rule)

        return self._prediction_vector(X, rule_individuals)
    
    @abstractmethod
    def _compute_rule(self, rule):
        raise NotImplementedError

    def _prediction_vector(self, X, rule_individuals):
        y_pred = np.zeros_like(X)
        for i in range(X.shape[0]):
            if X[i] in rule_individuals:
                y_pred[i] = 1
        return y_pred.astype(int)


class SimpleRulePredictionWrapper(RulePredictionWrapper):

    def __init__(self, trained_model, individuals, classes):
        super().__init__(trained_model, individuals, classes)

    def _compute_rule(self, rule):
        if len(rule.Classes) > 0:
            ind_set = set(self.individuals)
            for cls in rule.Classes:
                ind_set.intersection_update(set(cls.instances()))
            return ind_set
        else:
            return set([])
        

class HardRulePredictionWrapper(RulePredictionWrapper):

    def __init__(self, trained_model, individuals, classes):
        super().__init__(trained_model, individuals, classes)

    def _compute_rule(self, rule):

        if len(rule.Classes) > 0:
            ind_set = set([])

            for disj_cls in rule.Classes:
               
                if len(disj_cls.Classes) > 0:
                    conj_ind_set = set(self.individuals)
                    for conj_cls in disj_cls.Classes:
                        conj_ind_set.intersection_update(set(conj_cls.instances()))
                    ind_set.update(conj_ind_set)
            return ind_set
        else:
            return set([])

