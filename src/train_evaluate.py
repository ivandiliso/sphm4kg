"""
University of Bari Aldo Moro
@authors: Ivan Diliso, Nicola Fanizzi

# Intended Usage
python3 train_evaluate.py --onto ["lubm", "financial", "ntnames", "krkrzeroone"]

"""

import argparse
import pickle

import numpy as np
from owlready2 import *
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from MBM.models.MBNB import MBNB
from MBM.models.HB import (
    BernoulliMixtureVB,
    BernoulliMixtureEM,
    BernoulliMixtureSGD,
    HierachicalBernoulliMixture,
    TwoTierMixture
)
from MBM.models.rule_helpers import (
    HardRulePredictionWrapper,
    SimpleRulePredictionWrapper,
)
from MBM.ontology import Ontology
from utils import color, pretty_print
from sklearn.linear_model import LogisticRegression

# Arguments Parsing
################################################################################

onto_name_to_filename = {
    "lubm": "lubm.owl",
    "financial": "financial-abbrev.owl",
    "krkzeroone": "KRKZEROONE.owl",
    "ntnames": "NTNames.owl",
    "dbpedia" : "dbpedia_parsed.xml"
}

parser = argparse.ArgumentParser("Experiments Configurations")
parser.add_argument(
    "--onto",
    type=str,
    choices=["lubm", "financial", "krkzeroone", "ntnames", "dbpedia"],
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
MAX_ITER = 200
N_COMP = 5  
N_INIT = 10 
LR = 0.0001
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


# FEATURE CONSTRUCTION and SELECTION
################################################################################

X = ontology.features_to_matrix()
X_ind = np.array(ontology.individuals)

print("\nFeature Selection Phase")
print("Old Feature Matrix X: ", X.shape)
feature_selector = VarianceThreshold(threshold=0.1) 
X = feature_selector.fit_transform(X)

print("New Feature Matrix X: ", X.shape)
chosen_features_indices = feature_selector.get_support(indices=True)

print("")
print(f"# Chosen Features: {X.shape[1]}")
for i in chosen_features_indices:
    print(f"\t{i:5d} : {ontology.features[i]}")

owl_features = [ontology.features[i] for i in chosen_features_indices]


# PREDICTION TARGET ONTOLOGY
################################################################################


pretty_print("\nLoading Target Classes\n", color.CYAN)

ontology.load_targets(TARGET_FOLDER)


# LEARNING MODELS
################################################################################


batch_size = X.shape[0]

# MBNB
bernoulli_nb = MBNB() 

# HB_VB
mixture_h_vb = HierachicalBernoulliMixture(
    BernoulliMixtureVB, n_components=N_COMP, n_init=N_INIT, n_iter=MAX_ITER
)


# HB_EM
mixture_h_em = HierachicalBernoulliMixture(
    BernoulliMixtureEM, n_components=N_COMP, max_iter=MAX_ITER, tol=1e-3
)

# HB_GD
mixture_h_gd = HierachicalBernoulliMixture(
    BernoulliMixtureSGD,
    n_components=N_COMP,
    max_iter=MAX_ITER,
    learning_rate=LR,
    tol=1e-3,
    batch_size=batch_size,
)


logreg = LogisticRegression(C=0.01, penalty='l1', multi_class='multinomial', solver='saga', max_iter=200)
hlogreg = TwoTierMixture(BernoulliMixtureVB, n_components=5)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, criterion="log_loss")


pretty_print("\nPrediction models\n", color.CYAN)


models = {
    'MBNB'  : bernoulli_nb,
    'HB_VB' : mixture_h_vb,
    'HB_EM' : mixture_h_em,
    'HB_GD' : mixture_h_gd,
    "Tree": tree,
    "HLogReg": hlogreg,
    "LogREg": logreg
}


print(f"# Model Selected: {len(models.keys())}")

for model in models.keys():
    print(f"\t{color.RED}{model}{color.END} : {models[model]}")

y = ontology.target_to_vector(0, type="simple")


# TRAINING AND EVALUTAION
################################################################################


res_dict = {"simple": None, "hard": None}


for target_type in ["simple", "hard"]:

    pretty_print(
        f"\nStarting Traning Phase for {target_type.upper()} Targets\n", color.CYAN
    )

    targets = ontology.get_targets(target_type)
    averages = np.empty((len(models), 4, len(targets)))
    rule_averages = np.empty((len(models), 4, len(targets)))

    for target_id in range(len(targets)):

        models = {
            'MBNB'  : clone(bernoulli_nb),
            'HB_VB' : clone(mixture_h_vb),
            'HB_EM' : clone(mixture_h_em),
            'HB_GD' : clone(mixture_h_gd),
            "Tree": clone(tree),
            "HLogReg" : clone(hlogreg),
            "LogReg" : clone(logreg)
        }

        # Compute the target y vector

        target = targets[target_id]
        y = ontology.target_to_vector(target_id, target_type).astype(int)

        pretty_print(f"Training on Target {target}", color.GREEN)
        print(f"≡ {target.equivalent_to}")
        print(f"# Positive Istances {y.sum()}")

        for model_index, model_name in enumerate(models.keys()):

            pretty_print(f"Model: {model_name}", color.RED)

            model = models[model_name]
            scores = {}
            rule_scores = {}

            train_cv = StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=SEED
            )

            for train_index, test_index in train_cv.split(X, y):


                # Training Phase: Probabilistic Model
                # Splitting indivuals (both DL an Vectorized formats) into train and test

                X_train, X_test = X[train_index], X[test_index]
                X_ind_train, X_ind_test = X_ind[train_index], X_ind[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                scores.setdefault("P", []).append(
                    precision_score(
                        y_test,
                        y_pred,
                        labels=[1, 0],
                        average="weighted",
                        zero_division=1,
                    )
                )
                scores.setdefault("R", []).append(
                    recall_score(y_test, y_pred, labels=[1, 0], average="weighted")
                )
                scores.setdefault("F1", []).append(
                    f1_score(y_test, y_pred, average="weighted", zero_division=1)
                )
                scores.setdefault("AUC", []).append(
                    roc_auc_score(y_test, y_pred_proba, average="weighted")
                )

                # Axiom Extraction Phase with Threshold Optimization

                if model_name in {"HB_VB", "HB_EM", "HB_GD"}:
                    rule_wrapper = HardRulePredictionWrapper(
                        trained_model=model,
                        individuals=ontology.individuals,
                        classes=owl_features,
                    )
                    grid = ParameterGrid(
                        {
                            "theta_classes": [
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                            ],
                            "theta_clusters": [
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                            ],
                        }
                    )

                    best_params = None
                    best_params_score = 0
                    for params in grid:
                        param_score = accuracy_score(
                            y_train, rule_wrapper.predict(X_ind_train, **params)
                        )

                        if param_score > best_params_score:
                            best_params_score = param_score
                            best_params = params

                    y_rule_pred = rule_wrapper.predict(X_ind_test, **best_params)

                elif model_name in {"MBNB"}:
                    rule_wrapper = SimpleRulePredictionWrapper(
                        trained_model=model,
                        individuals=ontology.individuals,
                        classes=owl_features,
                    )
                    grid = ParameterGrid(
                        {
                            "theta_classes": [
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                            ],
                        }
                    )

                    best_params = None
                    best_params_score = 0
                    for params in grid:
                        param_score = accuracy_score(
                            y_train, rule_wrapper.predict(X_ind_train, **params)
                        )

                        if param_score > best_params_score:
                            best_params_score = param_score
                            best_params = params

                    y_rule_pred = rule_wrapper.predict(X_ind_test, **best_params)

                else:
                    y_rule_pred = y_pred

                rule_scores.setdefault("P", []).append(
                    precision_score(
                        y_test,
                        y_rule_pred,
                        labels=[1, 0],
                        average="weighted",
                        zero_division=1,
                    )
                )
                rule_scores.setdefault("R", []).append(
                    recall_score(y_test, y_rule_pred, labels=[1, 0], average="weighted")
                )
                rule_scores.setdefault("F1", []).append(
                    f1_score(y_test, y_rule_pred, average="weighted", zero_division=1)
                )
                rule_scores.setdefault("AUC", []).append(
                    roc_auc_score(y_test, y_rule_pred, average="weighted")
                )

            pretty_print("\tAverage Scores: Probabilistic Model", color.PURPLE)

            for s, score in enumerate(scores.keys()):
                mean = np.array(scores[score]).mean()
                std_dev = np.array(scores[score]).std()
                print("\t %.3f ± %.3f %s" % (mean, std_dev, score))
                averages[model_index][s][target_id] = mean

            pretty_print("\tAverage Scores: Deterministic Rule", color.PURPLE)

            for s, score in enumerate(rule_scores.keys()):
                mean = np.array(rule_scores[score]).mean()
                std_dev = np.array(rule_scores[score]).std()
                print("\t %.3f ± %.3f %s" % (mean, std_dev, score))
                rule_averages[model_index][s][target_id] = mean

    res_dict[target_type] = (averages, rule_averages)


with open(ONTO_NAME + ".bin", "wb") as f:
    pickle.dump(res_dict, f)


for target_type in ["simple", "hard"]:
    table = []
    headers = ["Dataset", "Model", "Type", "Precision", "Recall", "F1", "AUC"]

    averages, rule_averages = res_dict[target_type]

    for m, m_name in enumerate(models.keys()):
        for sel_score, sel_name in zip([averages, rule_averages], ["Prob", "Rule"]):

            row = [
                ontology.onto.name,
                m_name,
                sel_name,
                f"{sel_score[m][0].mean():1.3f} ± {sel_score[m][0].std():1.3f}",
                f"{sel_score[m][1].mean():1.3f} ± {sel_score[m][1].std():1.3f}",
                f"{sel_score[m][2].mean():1.3f} ± {sel_score[m][2].std():1.3f}",
                f"{sel_score[m][3].mean():1.3f} ± {sel_score[m][3].std():1.3f}",
            ]

            table.append(row)

    pretty_print(f"Results for the {target_type.upper()} Targets", color.RED)
    print(tabulate(table, headers=headers, tablefmt="github"))
