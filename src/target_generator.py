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
MAX_DISJUNCTION_TERMS = 6
MAX_CONJUNCTION_TERMS = 3
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
    (c, set(c.instances())) for c in ontology.features if len(set(c.instances())) > 0
]

num_ind = len(set(ontology.individuals))
MAX_IND = int((num_ind / 2) * (1 + MIN_PERC_EXAMPLES))
MIN_IND = int((num_ind / 2) * (1 - MIN_PERC_EXAMPLES))


print("\nSIMPLE Target Generation Phase\n##############################\n")

print(
    f"Total number of individuals {num_ind}, will need targets with at lead {MIN_IND} to {MAX_IND} individuals"
)

"""
for i in range(len(valid_classes)):
    c, instances = valid_classes[i]
    print(i, c, len(instances), c.equivalent_to)
"""

num_valid_classes = len(valid_classes)
print(
    f"We have a total of {num_valid_classes} class for targers (features with at least 1 individuals)"
)

a = input()

simple_targets = set()
seen_ids = set()
uniqueness_score = 0
counter = 1
uniqueness_treshold = 4

simple_targets_class_y = []


target_onto_name = "http://www.example.org/" + ontology.onto.name + "/targets"
target_onto = get_ontology(target_onto_name)

with target_onto:
    Target = types.new_class("Target", (Thing,))


while len(simple_targets) < N_SIMPLE_TARGETS:

    disjuction_terms_ids = np.random.choice(
        range(num_valid_classes),
        np.random.randint(2, MAX_DISJUNCTION_TERMS + 1),
        replace=False,
    )
    disjuction_terms_ids = tuple(sorted(disjuction_terms_ids.tolist()))
    disjution = set()
    for term_id in disjuction_terms_ids:
        c, instances = valid_classes[term_id]
        disjution = disjution.union(instances)

    print(
        f"Trying for element {len(simple_targets)} # {counter} - Try {disjuction_terms_ids} of len {len(disjution)}"
    )

    if (
        len(disjution) < MAX_IND
        and len(disjution) > MIN_IND
        and disjuction_terms_ids not in simple_targets
    ):
        uniqueness_offset = 0

        for element in disjuction_terms_ids:
            if element in seen_ids:
                uniqueness_offset += 1

        if (uniqueness_offset + uniqueness_score) < uniqueness_treshold:
            uniqueness_score += uniqueness_offset

            simple_targets.add(disjuction_terms_ids)

            y = np.zeros_like(ontology.individuals)
            for i, ind in enumerate(ontology.individuals):
                if ind in disjution:
                    y[i] = 1

            with target_onto:

                t_name = "Simple_Class_" + str(len(simple_targets))
                candidate = types.new_class(t_name, (Target,))

                owl_class = None
                for term in disjuction_terms_ids:
                    c, instances = valid_classes[term]
                    if owl_class is None:
                        owl_class = c
                    else:
                        owl_class = owl_class | c

                candidate.equivalent_to = [owl_class]
                simple_targets_class_y.append((t_name, y))

            counter = 1
            for element in disjuction_terms_ids:
                seen_ids.add(element)
            else:
                counter += 1
    else:
        counter += 1

print(f"Generated Simple Targets with UNIQUENESS SCORE {uniqueness_score}")
for target in simple_targets:
    print(target)

for c, y in simple_targets_class_y:
    print(c, y, np.sum(y))

for cls in target_onto.classes():
    print(cls, cls.equivalent_to)

with open(TARGET_FOLDER + f"/{ONTO_NAME}_simple_targets.pkl", "wb") as f:
    pickle.dump(simple_targets_class_y, f)



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


while len(hard_targets) < N_HARD_TARGETS:

    break_check = False
    disjunction_terms_ids = set()
    for _ in range(np.random.randint(2, MAX_DISJUNCTION_TERMS + 1)):

        conjunction_term_ids = np.random.choice(
            range(num_valid_classes),
            np.random.randint(2, MAX_CONJUNCTION_TERMS + 1),
            replace=False,
        )
        conjunction_term_ids = tuple(sorted(conjunction_term_ids.tolist()))
        disjunction_terms_ids.add(conjunction_term_ids)

    print(
        f"Trying for element {len(hard_targets)} # {counter} - Try {disjunction_terms_ids}"
    )

    disjunction = set()
    for disj_term in disjunction_terms_ids:
        conjunction = set()
        for i, conj_term in enumerate(disj_term):
            c, instances = valid_classes[conj_term]
            # print(f"doing term {conj_term} in of {disj_term} with {len(instances)}")
            if i == 0:
                conjunction = set(instances)
            else:
                conjunction = conjunction.intersection(instances)
        # print(f"Conjuction has len {len(conjunction)}")
        # print("\n")
        if len(conjunction) == 0:
            break_check = True

        disjunction = disjunction.union(conjunction)

    if break_check:
        print("THIS WILL BE EXLUDED, BREAK")
        counter += 1
        continue

    print(seen_tuples)
    print(disjunction_terms_ids)

    if len(seen_tuples.intersection(disjunction_terms_ids)) > 0:
        print("Already seen tuple, BREAK")
        counter += 1
        continue

    print("Found disjuntion of len ", len(disjunction))

    if len(disjunction) < MAX_IND and len(disjunction) > MIN_IND:
        hard_targets.add(tuple(disjunction_terms_ids))
        counter = 1

        y = np.zeros_like(ontology.individuals)
        for i, ind in enumerate(ontology.individuals):
            if ind in disjunction:
                y[i] = 1

        with target_onto:

            t_name = "Hard_Class_" + str(len(hard_targets))
            candidate = types.new_class(t_name, (Target,))

            owl_class = None
            for disj_term in disjunction_terms_ids:
                seen_tuples.add(disj_term)
                print(seen_tuples)
                conj_owl = None
                for conj_term in disj_term:
                    c, instances = valid_classes[conj_term]
                    if conj_owl is None:
                        conj_owl = c
                    else:
                        conj_owl = conj_owl & c

                if owl_class is None:
                    owl_class = conj_owl
                else:
                    owl_class = owl_class | conj_owl

            candidate.equivalent_to = [owl_class]
            hard_targets_class_y.append((t_name, y))

    else:
        counter += 1


for target in hard_targets:
    print(target)

for c, y in hard_targets_class_y:
    print(c, y, np.sum(y))

for cls in target_onto.classes():
    print(cls, cls.equivalent_to)

with open(TARGET_FOLDER + f"/{ONTO_NAME}_hard_targets.pkl", "wb") as f:
    pickle.dump(hard_targets_class_y, f)


print("\nSAVING TARGET ON DISK AS PICKLES\n")

filename = TARGET_FOLDER + "/" + ontology.onto.name + "-t.nt"
target_onto.save(file=filename)
target_onto.destroy()
