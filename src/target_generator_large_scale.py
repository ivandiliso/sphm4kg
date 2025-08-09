"""
University of Bari Aldo Moro
@authors: Ivan Diliso, Nicola Fanizzi

# Intended Usage
python3 target_generator.py --onto ["lubm", "financial", "ntnames", "krkrzeroone"]

"""

import pickle
import types
from random import randrange, seed

import numpy as np
from owlready2 import *
import argparse
from MBM.ontology import Ontology
from utils import SimpleLogger

# Arguments Parsing
################################################################################

onto_name_to_filename = {
    "lubm": "lubm.owl",
    "financial": "financial-abbrev.owl",
    "krkzeroone": "KRKZEROONE.owl",
    "ntnames": "NTNames.owl",
    "dbpedia" : "dbpedia_parsed.xml",
    "yago" : "yago_parsed.xml"
}

parser = argparse.ArgumentParser("Experiments Configurations")
parser.add_argument(
    "--onto",
    type=str,
    choices=["lubm", "financial", "krkzeroone", "ntnames", "dbpedia", "yago"],
    required=True,
)

args = parser.parse_args()
logger = SimpleLogger()


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

MIN_PERC_EXAMPLES = 0.30
MAX_DISJUNCTION_TERMS = 3
MAX_CONJUNCTION_TERMS = 2
N_SIMPLE_TARGETS = 10
N_HARD_TARGETS = 0


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

[!] Only in this phase the data propriety are NOT used 
"""

ontology.compute_features(object_props=True, data_props=False)
ontology.features_summary()


# INDIVIDUAL CONSTRUCTION
################################################################################


ontology.extract_individuals()

print(f"# Individuals INFERRED: {len(ontology.individuals)}")


logger.start("Syncinc Reasoner on onto")
sync_reasoner_pellet(debug=0)
logger.end()


valid_classes = [
    (c, set(c.instances())) for c in ontology.features if len(set(c.instances())) > 200
]

num_ind = len(set(ontology.individuals))
MAX_IND = int((num_ind / 2) * (1 + MIN_PERC_EXAMPLES))
MIN_IND = int((num_ind / 2) * (1 - MIN_PERC_EXAMPLES))

conj_min_elements = int(MIN_IND / (MAX_DISJUNCTION_TERMS+1))
conj_max_elements = int(MAX_IND / (MAX_DISJUNCTION_TERMS+1))



print(
    f"Total number of individuals {num_ind}, will need targets with at lead {MIN_IND} to {MAX_IND} individuals"
)

print(
    f"Each conjunctiver term will required to have from {conj_min_elements} to {conj_max_elements}"
)



num_valid_classes = len(valid_classes)
print(
    f"We have a total of {num_valid_classes} class for targers (features with at least 1 individuals)"
)


print("\nHARD Target Generation Phase\n##############################\n")


hard_targets = set()
seen_ids = set()
simple_targets = set()
seen_ids = set()
uniqueness_score = 0
counter = 1
uniqueness_treshold = 4
hard_targets_class_y = []
seen_tuples = set()



candidates_in_disjunction = set()
max_candidates = 50
final_candidates = list()

while len(candidates_in_disjunction) < max_candidates:

    break_check = False
    conjunction_term_ids = np.random.choice(
        range(num_valid_classes),
        np.random.randint(2, MAX_CONJUNCTION_TERMS + 1),
        replace=False,
    )
    conjunction_term_ids = tuple(sorted(conjunction_term_ids.tolist()))


    print(
        f"Trying for element {len(candidates_in_disjunction)} # {counter} - Try {conjunction_term_ids}"
    )


   
    conjunction = set()
    for i, conj_term in enumerate(conjunction_term_ids):
        c, instances = valid_classes[conj_term]
        if i == 0:
            conjunction = set(instances)
        else:
            conjunction = conjunction.intersection(instances)
    if len(conjunction) == 0:
        print("Was empty")
        break_check = True



    if break_check:
        print("THIS WILL BE EXLUDED, BREAK")
        counter += 1
        continue

    print(seen_tuples)

    if len(seen_tuples.intersection(conjunction_term_ids)) > 0:
        print("Already seen tuple, BREAK")
        counter += 1
        continue

    print("Found conjunction of len ", len(conjunction))

    if len(conjunction) > 240:
        candidates_in_disjunction.add(tuple(conjunction_term_ids))
        final_candidates.append((len(conjunction),tuple(conjunction_term_ids)))
        counter = 1


print("At last")
print(candidates_in_disjunction)


for l, candidate in final_candidates:
    out = ""
    for c in candidate:
        out += str(valid_classes[c][0].iri) + " <-> "
    print(l, out)


