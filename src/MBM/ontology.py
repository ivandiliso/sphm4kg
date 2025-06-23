"""
University of Bari Aldo Moro

@author: Ivan Diliso, Nicola Fanizzi
"""

import os
import owlready2
from owlready2 import *
import sys
import numpy as np
from utils import SimpleLogger

UNKNOWN = 0
POSITIVE = 1


class Ontology:
    def __init__(
        self,
        onto_file: str,
        onto_folder: str = "onto",
        java_memory: int = 4000,
        random_seed: int = 42,
    ):

        self.ontology_file = onto_file
        self.onto_folder = onto_folder
        self.java_memory = java_memory
        self.random_seed = random_seed

        self.onto = None  # This will hold the loaded ontology object
        self.features = None  # This will hold the loaded features
        self.features_complement = (
            None  # This will hold the loaded features complements
        )

        self._set_java()

    def _set_java(self):
        """Set Owlready2 to use the system Java EXE (Using Java Home) and the specified memory"""

        if sys.platform == "linux" or sys.platform == "linux32":
            print("Running JAVA in Linux")
            owlready2.JAVA_EXE = "/usr/lib/jvm/default-java/bin/java"
        elif sys.platform == "darwin":
            print("Running JAVA in MacOS")
            java_home_path = os.popen("/usr/libexec/java_home").read().strip()
            owlready2.JAVA_EXE = os.path.join(java_home_path, "bin", "java")

        owlready2.reasoning.JAVA_MEMORY = self.java_memory

        print(f"Owlready2 will use Java from: {owlready2.JAVA_EXE}")
        print(f"Owlready2 Java Memory set to: {owlready2.reasoning.JAVA_MEMORY} MB")

    def load_ontology(self, reload: bool = True):
        """Load the specified OWL ontology

        Args:
            reload (bool, optional): Reload the ontology if already loaded. Defaults to True.
        """
        onto_path.append(self.onto_folder)
        self.onto = get_ontology(self.ontology_file).load(reload=reload)

        self.classes = list(self.onto.classes())
        self.individuals = list(self.onto.individuals())
        self.oprops = list(self.onto.object_properties())
        self.dprops = list(self.onto.data_properties())

        print(f"Ontology '{self.ontology_file}' LOADED")

    def summary(self):
        """Prints summary of the loaded ontology"""
        if self.onto is None:
            print("Ontology not loaded yet. Call load_ontology() first.")
            return

        print("")
        print(f"Base IRI: {self.onto.base_iri}")
        print(f"# Classes: {len(list(self.onto.classes()))}")

        onto = self.onto

        for i, cls in enumerate(onto.classes()):
            print(f"\t{i:5d} : {cls} is_a {cls.is_a} equivalent_to {cls.equivalent_to}")

        print(f"# Disjoint Classes: {len(list(self.onto.disjoint_classes()))}")

        for i, dcls in enumerate(list(self.onto.disjoint_classes())):
            print(f"\t{i:5d} : {dcls}")

        print("# Obj-props: %d " % (len(self.oprops)))

        for j, obj_property in enumerate(self.oprops):
            print(
                "\t%5d: %s (%s >> %s)"
                % (j, obj_property, obj_property.domain, obj_property.range)
            )

        print("# Data-props: %d " % (len(self.dprops)))

        for j, d_property in enumerate(self.dprops):
            print(
                "\t%5d: %s (%s >> %s)"
                % (j, d_property, d_property.domain, d_property.range)
            )

        print("# Individuals ASSERTED: %d \n" % (len(self.individuals),))

    def compute_features(self, object_props=True, data_props=True):

        # CLASSES
        self.features = self.classes.copy()

        # OBJECT PROPRIETIES RANGE AND DOMAIN TO FEATURE (DOMAIN EXCLUDED)

        if object_props:
            with self.onto:
                for p in range(len(self.oprops)):

                    # Domain and range of the propriety
                    dom_p = (
                        self.oprops[p].domain
                        if not (self.oprops[p].domain == [])
                        else [Thing]
                    )
                    ran_p = (
                        self.oprops[p].range
                        if not (self.oprops[p].range == [])
                        else [Thing]
                    )

                    range_new_feature = types.new_class(
                        "Some_" + self.oprops[p].name + "_" + "range", (Thing,)
                    )

                    range_new_feature.equivalent_to = [self.oprops[p].some(ran_p[0])]
                    self.features.append(range_new_feature)

        if data_props:
            with self.onto:
                for p in range(len(self.dprops)):

                    dom_p = (
                        self.dprops[p].domain
                        if not (self.dprops[p].domain == [])
                        else [Thing]
                    )
                    ran_p = (
                        self.dprops[p].range
                        if not (self.dprops[p].range == [])
                        else [Thing]
                    )
                    new_feature = types.new_class(
                        "Some_" + self.dprops[p].name + "_" + "range", (Thing,)
                    )  # dom_p[0]
                    new_feature.equivalent_to = [self.dprops[p].some(ran_p[0])]
                    self.features.append(new_feature)

        self.features_complement = []

        with self.onto:
            for a_class in self.features:
                complement = types.new_class("Non_" + a_class.name, (Thing,))
                complement.equivalent_to = [Not(a_class)]
                self.features_complement.append(complement)

            sync_reasoner_pellet(debug=0)

    def features_to_matrix(self):

        X = np.full((len(self.individuals), len(self.features)), UNKNOWN)

        for f in range(len(self.features)):
            for ind in set(self.features[f].instances()):
                i = self.individuals.index(ind)
                X[i, f] = POSITIVE

        return X

    def load_targets(self, target_folder):

        filename = target_folder + "/" + self.onto.name + "-t.nt"
        target_onto = get_ontology(filename).load()

        print("")
        self.simple_targets = np.array(
            sorted(
                set(target_onto.search(iri="*#Simple_Class*")), key=(lambda x: x.name)
            )
        )
        self.hard_targets = np.array(
            sorted(set(target_onto.search(iri="*#Hard_Class*")), key=(lambda x: x.name))
        )

        print(f"# SIMPLE Disjunctive Classes: {len(self.simple_targets)}")
        for i, target in enumerate(self.simple_targets):
            print(
                f"\t{i:02d} : {target} ({int(self.target_to_vector(i, type="simple").sum())}) ≡ {target.equivalent_to}"
            )
        print("")
        print(f"# HARD Disjunction of Conjunctive Classes: {len(self.hard_targets)}")
        for i, target in enumerate(self.hard_targets):
            print(
                f"\t{i:02d} : {target} ({int(self.target_to_vector(i, type="hard").sum())}) ≡ {target.equivalent_to}"
            )

    def _conjunctive_set(self, target):

        ind_set = set(self.individuals)

        for elem in target.Classes:
            ind_set.intersection_update(set(elem.instances()))

        return ind_set

    def ext_target_to_vector(self, target, type: str):
        """
        type can be simple or hard, different computation
        """

        y = np.zeros_like(self.individuals)
        ind_set = set()

        if type == "simple":
            ind_set = self._conjunctive_set(target)
        elif type == "hard":
            ind_set = self._hard_feature_ind_set(target)
        else:
            raise Exception("Wrong target type, must be in 'hard' or 'simple'")

        for i, ind in enumerate(self.individuals):
            if ind in ind_set:
                y[i] = 1

        return y.astype(int)

    def target_to_vector(self, target_id, type: str):
        """
        type can be simple or hard, different computation
        """

        y = np.zeros_like(self.individuals)
        ind_set = set()

        if type == "simple":
            ind_set = self._simple_feature_ind_set(
                self.simple_targets[target_id].equivalent_to[0]
            )
        elif type == "hard":
            ind_set = self._hard_feature_ind_set(
                self.hard_targets[target_id].equivalent_to[0]
            )
        else:
            raise Exception("Wrong target type, must be in 'hard' or 'simple'")

        for i, ind in enumerate(self.individuals):
            if ind in ind_set:
                y[i] = 1

        return y.astype(int)

    def get_targets(self, type: str):
        if type == "simple":
            return self.simple_targets
        elif type == "hard":
            return self.hard_targets
        else:
            raise Exception("wrong type")

    def _simple_feature_ind_set(self, target):

        ind_set = set()

        for cls in target.Classes:
            ind_set.update(set(cls.instances()))

        return ind_set

    def _hard_feature_ind_set(self, target):

        ind_set = set()

        for disj_cls in target.Classes:
            conj_set = set(self.individuals)
            for conj_cls in disj_cls.Classes:
                conj_set.intersection_update(set(conj_cls.instances()))

            ind_set.update(conj_set)

        return ind_set

    def features_summary(self):
        print(f"# Features: {len(self.features)}")
        for i, feat in enumerate(self.features):
            print(f"\t{i:5d} : {feat} ≡ {feat.equivalent_to}")

    def extract_individuals(self):

        for i, f in enumerate(self.features):
            self.individuals = self.individuals + list(f.instances())

        # Sort individuals by name and get only uniques
        self.individuals = sorted(set(self.individuals), key=lambda an_ind: an_ind.name)
