from rdflib import Graph, RDF, RDFS, OWL, Namespace
from rdflib.namespace import split_uri
from rdflib.term import URIRef
from pathlib import Path
import pickle
from tqdm import tqdm


file = Path().cwd().absolute() / "data/raw/yago4-20/yago-wd-facts.nt"
num_lines = sum(1 for _ in open(file))
print(num_lines)

entities = set()

with open(file, "r") as f:
    for i in tqdm(range(num_lines)):
        line = f.readline().split('\t')
        s = line[0].strip("<>")
        o = line[2].strip("<>")
        
        entities.add(s)
        entities.add(o)

with open("yago_individuals_uri.bin", "wb") as f:
    pickle.dump(entities, f)


