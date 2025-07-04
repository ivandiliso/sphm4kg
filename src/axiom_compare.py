"""
University of Bari Aldo Moro
@authors: Ivan Diliso, Nicola Fanizzi

# Intended Usage
python3 axiom_compare.py --onto ["lubm", "financial", "ntnames", "krkrzeroone"]

"""

import argparse
import os
import pickle
import types
from random import seed

import numpy as np
import pandas as pd
import scikit_posthocs as sp
import wittgenstein as lw
from imblearn.metrics import geometric_mean_score
from owlready2 import *
from owlready2 import (
    Not,
    Thing,
    get_ontology,  # , reasoning, IRIS
    onto_path,
    reasoning,
    sync_reasoner_pellet,
)
from scipy.stats import uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    cross_validate,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from MBM.models.HB import (
    BernoulliMixtureVB,
    BernoulliMixtureEM,
    BernoulliMixtureSGD,
    HierachicalBernoulliMixture,
)
from MBM.models.MBNB import MBNB
from MBM.models.rule_helpers import (
    HardRulePredictionWrapper,
    SimpleRulePredictionWrapper,
)
from MBM.ontology import Ontology
from utils import SimpleLogger, color, pretty_print

logger = SimpleLogger()

# Arguments Parsing
################################################################################

onto_name_to_filename = {
    "lubm": "lubm.owl",
    "financial": "financial-abbrev.owl",
    "krkzeroone": "KRKZEROONE.owl",
    "ntnames": "NTNames.owl",
}

parser = argparse.ArgumentParser("Experiments Configurations")
parser.add_argument(
    "--onto",
    type=str,
    choices=["lubm", "financial", "krkzeroone", "ntnames"],
    required=True,
)

args = parser.parse_args()


# PARAMS
################################################################################


SEED = 42
N_SPLITS = 5
JAVA_MEMORY = 4000
TEST_PORTION = 0.3
UNLABEL = -1
ONTO_FOLDER = "../data/onto"
TARGET_FOLDER = "../data/onto_target"
ONTO_FILE = onto_name_to_filename[args.onto]
ONTO_NAME = ONTO_FILE.split(".")[0]


# ONTOLOGY LOADING
################################################################################


ontology = Ontology(
    onto_file=ONTO_FILE,
    onto_folder=ONTO_FOLDER,
    java_memory=JAVA_MEMORY,
    random_seed=SEED,
)


ontology.load_ontology(reload=True)
ontology.summary()

onto = ontology.onto


# FEATURE CONSTRUCTION
################################################################################

"""
Here we will also compute the class complement and sync the pellet reasoner
This will be useful in the individual computation next phase
"""

ontology.compute_features(object_props=True, data_props=True)
ontology.features_summary()


# INDIVIDUAL CONSTRUCTION
################################################################################


ontology.extract_individuals()

print(f"# Individuals INFERRED: {len(ontology.individuals)}")


X = ontology.features_to_matrix()
X_ind = np.array(ontology.individuals)
owl_features = ontology.features


# PREDICTION TARGET ONTOLOGY
################################################################################

t = ontology.features

"""
target = Or([
    And([t[20], t[63]]),
    And([t[25], t[63]]),
    And([t[36], t[64]])
    ])
"""


"""
lubm.Faculty 35
lubm.Some_doctoralDegreeFrom_range 35
lubm.Some_teacherOf_range 35
35


lubm.GraduateStudent & lubm.Person & lubm.Student & lubm.Some_advisor_range & lubm.Some_degreeFrom_range & lubm.Some_memberOf_range & lubm.Some_takesCourse_range & lubm.Some_undergraduateDegreeFrom_range
lubm.GraduateStudent 119
lubm.Some_advisor_range 168
lubm.Some_memberOf_range 437
lubm.Some_takesCourse_range 402
119


"""


target = Or([And([t[23], t[66]]), And([t[23], t[47]]), And([t[23], t[51]])])


"""
target = Or([
    And([t[15], t[65]]),
    And([t[3], t[67]]),
    And([t[5], t[49]])
    ])
"""

print(target)
ind_set = set([])

for disj in target.Classes:
    print(disj)
    c_set = set(ontology.individuals)
    for conj in disj.Classes:
        print(conj, len(set(conj.instances())))
        c_set.intersection_update(set(conj.instances()))
    print(len(c_set))
    ind_set.update(c_set)


y = np.zeros_like(ontology.individuals)


for i, ind in enumerate(ontology.individuals):
    if ind in ind_set:
        y[i] = 1


y = y.astype(int)
print(y.sum())


# LEARNING MODELS
################################################################################

MAX_ITER = 200
N_COMP = 5  # Random number, may will change with new optimal one in grid_search
N_INIT = 10  # Random number, may will change with new optimal one in grid_search
lr = 0.0001
batch_size = X.shape[0]

bernoulli_nb = MBNB()
mixture_h_vb = HierachicalBernoulliMixture(
    BernoulliMixtureVB, n_components=N_COMP, n_init=N_INIT, n_iter=MAX_ITER
)
mixture_h_em = HierachicalBernoulliMixture(
    BernoulliMixtureEM, n_components=N_COMP, max_iter=MAX_ITER, tol=1e-3
)
mixture_h_gd = HierachicalBernoulliMixture(
    BernoulliMixtureSGD,
    n_components=N_COMP,
    max_iter=MAX_ITER,
    learning_rate=lr,
    tol=1e-3,
    batch_size=batch_size,
)


# TRAINING AND EVALUTAION
################################################################################


models = {
    "MBNB": clone(bernoulli_nb),
    "HB_VB": clone(mixture_h_vb),
    "HB_EM": clone(mixture_h_em),
    "HB_GD": clone(mixture_h_gd),
}


for model_index, model_name in enumerate(models.keys()):

    pretty_print(f"Model: {model_name}", color.RED)

    model = models[model_name]
    model.fit(X, y)

    if model_name in {"HB_VB", "HB_EM", "HB_GD"}:
        rule_wrapper = HardRulePredictionWrapper(
            trained_model=model, individuals=ontology.individuals, classes=owl_features
        )
        grid = ParameterGrid(
            {
                "theta_classes": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "theta_clusters": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )

        best_params = None
        best_params_score = 0
        for params in grid:
            param_score = accuracy_score(y, rule_wrapper.predict(X_ind, **params))

            if param_score > best_params_score:
                best_params_score = param_score
                best_params = params

        print(best_params, best_params_score)
        rule = model.extract_rule(owl_features, **best_params)

        for disj in rule.Classes:
            print(disj)
            c_set = set(ontology.individuals)
            for conj in disj.Classes:
                print(conj, len(set(conj.instances())))
                c_set.intersection_update(set(conj.instances()))
            print(len(c_set))

    elif model_name in {"MBNB"}:
        rule_wrapper = SimpleRulePredictionWrapper(
            trained_model=model, individuals=ontology.individuals, classes=owl_features
        )
        grid = ParameterGrid(
            {
                "theta_classes": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )

        best_params = None
        best_params_score = 0
        for params in grid:
            param_score = accuracy_score(y, rule_wrapper.predict(X_ind, **params))

            if param_score > best_params_score:
                best_params_score = param_score
                best_params = params

        print(best_params, best_params_score)
        rule = model.extract_rule(owl_features, **best_params)

    print(rule)
