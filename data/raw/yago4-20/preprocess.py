from rdflib import Graph, RDF, RDFS, OWL, Namespace
from rdflib.namespace import split_uri
from rdflib.term import URIRef
from pathlib import Path
import pickle
import csv
import ast

def safe_split_uri(uri):
    uri = str(uri)
    if "/resource/" in uri:
        return uri.split("/resource/")[-1]
    elif "/property/" in uri:
        return uri.split("/property/")[-1]
    elif "/ontology/" in uri:
        return uri.split("/ontology/")[-1]
    else:
        # Fallback: last fragment
        return uri.split("#")[-1].split("/")[-1]
    








home_path = Path().cwd().absolute() / "data/raw/yago4-20"



with open(home_path / "entities.csv", "r") as f:
    csv.reader(f, delimiter=",")



yago_class = Graph()
yago_class.parse(home_path / "yago-wd-class.nt")

yago_schema = Graph()
yago_schema.parse(home_path / "yago-wd-schema.nt")


classes = set(yago_class.subjects(RDF.type, RDFS.Class)) | set(yago_class.subjects(RDF.type, OWL.Class))
properties = set(yago_schema.subjects(RDF.type, OWL.ObjectProperty))

print(len(classes))
print(len(properties))


with open(home_path / "yago_individuals_uri.bin", "rb") as f :
    entities = pickle.load(f)

print(len(entities))


everything = classes | properties | entities
print(len(everything))

lookup_table = {safe_split_uri(uri):uri for  uri in everything}





out_onto = Graph()
EX = Namespace("http://example.org/ontology#")
out_onto.bind("ex", EX)
out_onto.add((URIRef("http://example.org/ontology"), RDF.type, OWL.Ontology))


final_entities = set()
final_proprieties = set()
final_classes = set()

with open(home_path / "triples.txt", "r") as f:
    for line in f:

        s_label, p_label, o_label = line.strip().split("\t")
      
        s_uri = URIRef(lookup_table.get(s_label))
        p_uri = URIRef(lookup_table.get(p_label))
        o_uri = URIRef(lookup_table.get(o_label))

        if s_uri and p_uri and o_uri:
            out_onto.add((s_uri, RDF.type, OWL.NamedIndividual))
            out_onto.add((o_uri, RDF.type, OWL.NamedIndividual))
            out_onto.add((p_uri, RDF.type, OWL.ObjectProperty))
            out_onto.add(((s_uri, p_uri, o_uri)))

            final_entities.add(s_uri)
            final_entities.add(o_uri)
            final_proprieties.add(p_uri)

with open(home_path / "entities.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader)
    for row in reader:
        e = URIRef(lookup_table[row[0]])
        cs = ast.literal_eval(row[1])
        for c in cs:
            out_onto.add((URIRef(c), RDF.type, OWL.Class))
            out_onto.add((URIRef(c), RDFS.subClassOf, OWL.Thing))
            out_onto.add((e, RDF.type, URIRef(c)))
            final_classes.add(URIRef(c))


for s, o in yago_class.subject_objects(RDFS.subClassOf):
    if (s in final_classes) and (o in final_classes):
        out_onto.add((s, RDFS.subClassOf, o))



for obj_uri in final_proprieties:
    domain_ = list(yago_schema.objects(obj_uri, RDFS.domain))
    range_ = list(yago_schema.objects(obj_uri, RDFS.range))
    
    for dom_uri in domain_:
        if dom_uri in final_classes:
            out_onto.add((obj_uri, RDFS.domain, dom_uri))

    for range_uri in range_:
        if range_uri in final_classes:
            out_onto.add((obj_uri, RDFS.range, range_uri))



out_onto.serialize("yago_parsed.xml", format="xml")
