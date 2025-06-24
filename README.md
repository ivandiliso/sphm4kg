# Learning Interpretable Probabilistic Models and Schema Axioms for Knowledge Graphs

[![DOI](https://zenodo.org/badge/1005798178.svg)](https://doi.org/10.5281/zenodo.15708405)
![GitHub License](https://img.shields.io/github/license/ivandiliso/sphm4kg)
![Python Version](https://img.shields.io/badge/python-3.12.8%2B-blue)

## Folder Structure

```
📁 data             -> Ontologies (.owl) and their targets (.nt)
    📁 onto     
    📁 onto_target                
📁 out              -> Pickled output of models metrics on each target    
📁 src              -> Source code
    📁 MBM          -> Code regarding models, rule extraction and wrappers
```

## Source Code Structure

```
📁 src/HBM/models
    📄 HB.py                -> Hierarchical Model with Variational Bayes, EM and Gradient Descent, unified abstraction.
    📄 MBNB.py              -> Multivariate Bernouelli Naive Bayes Model
    📄 rule_helpers.py      -> Model wrappers for axiom extraction

📁 src/HBM
    📄 ontology.py          -> Ontology loading utility. Feature matrix contruction. Mapping from symbolic to vectorized formats

📁 src/
    📄 target_generator.py  -> Artificial disjuctinve problem generator
    📄 train_evaluate.py    -> Train and evaluation script (all models and all targets of a selected dataset)
    📄 axiom_compare.py     -> Utility to compare target and extracted axioms
    📄 utils.py             -> Logging and printing utils
```

## Project Details

### Requirements

The project has been developed using Python 3.12.8, a pip freeze of the packages used are available in the `requirements.txt` file.

### How to run experiments?

The `target_evaluate.py` Python code can be run with arguments, specifyng the dataset to be used, in order to ensure correct execution, execute the file in the src directory:

> ⚠ The dataset loading utility uses Owlready2 to automatically load the .owl files and run the Pellet reasoneer. The Owlready2 implementation requires specyfing the system JAVA_EXE path. This is handled automatically for Mac and Linux users, the system used during the experimentations (if using the standard location for the JAVA executibles) for Windows user, we suggest to check if some modification of `src/MBN/ontology.py _set_java()` is necessary

```bash
cd /path_to_project/sphm4kg/src
python3 train_evaluate.py --onto ("lubm", "financial", "ntnames", "krkrzeroone") # Choose one
```

### Target Concept Ontologies

The `onto_name.nt` in `data/onto_target` refer to the artificially created disjucntive targets created for each ontology in oder to assess the probabilistic model prediction capabilities. These files contain two types of targets:

- Simple_Class_X: Target defined as the disjuction of simple concepts
- Hard_Class_X: Target defined as the disjuction of conjunctions of simple concepts

To load them without using the provided utility, this code can be used (lubm in this example):

```python
from owlready2 import *

target_onto = get_ontology("./data/onto_target/lubm-t.nt").load()

simple_targets = set(target_onto.search(iri='*#Simple_Class*'))
hard_targets = set(target_onto.search(iri='*#Hard_Class*'))
```

### Feature Names

When working with the provided utility, you can see classes defined as:

- `namespace.ClassNaname`: These refer to a class defined in the ontology
- `namespace.Some_relationName_range`: These refer to existential restrictions on the relation range, formally defined as: 

$$ 
SomerealationNamerange \equiv \exists relationName.Range(relationName)
$$


### Hyperparameters Settings

| Model | Parameters and Ranges |
| - | - | 
| $\texttt{MBNB}$ | `{}`|
| $\texttt{HB}_{\texttt{VB}}$ | `{n_components = 5, n_init = 10, n_iter = 200}`| 
| $\texttt{HB}_{\texttt{EM}}$ | `{n_components = 5, max_iter = 200, tol=1e-3}` | 
| $\texttt{HB}_{\texttt{GD}}$ | `{n_components = 5, max_iter = 200, tol=1e-3, learning_rate=1e-4}`|
| $\texttt{Tree}$ |  `{max_depth=3, min_samples_leaf=10, criterion="log_loss"}` | 
| $\texttt{LogReg}$ | `{C=0.01, penalty='l1', solver='saga', max_iter=200}` |  
| $\texttt{HLogReg}$ | `{n_components = 5, n_init = 10, n_iter = 200, C=0.01, penalty='l1', solver='saga', max_iter=200}` |  
| $\texttt{AxiomWrapper}$ | `{theta_u : (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), theta_p : (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}`|  

