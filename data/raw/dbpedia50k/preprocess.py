from rdflib import Graph, RDF, RDFS, OWL, Namespace
from rdflib.namespace import split_uri
from rdflib.term import URIRef
from pathlib import Path

# http://yago-knowledge.org/resource/

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


home_path = Path().cwd().absolute() / "src/temp"

dbpedia_full = Graph()
dbpedia_full.parse(home_path / "db50.owl")

dbpedia_classes = Graph()
dbpedia_classes.parse(home_path / "db50_reasoned.owl")


classes = set(dbpedia_full.subjects(RDF.type, RDFS.Class)) | set(dbpedia_full.subjects(RDF.type, OWL.Class))
properties = set(dbpedia_full.subjects(RDF.type, OWL.ObjectProperty))
entities = set()

for s, p, o in dbpedia_full:
    if s not in classes and s not in properties:
        entities.add(s)
    if (o not in classes and o not in properties
        and not isinstance(o, str)):  # exclude literals
        entities.add(o)

with open("luigi.txt", "w") as f:
    f.write(str(entities))

test = URIRef('http://dbpedia.org/resource/Princeton,_New_Jersey')



everything = classes | properties | entities
print(len(everything))

lookup_table = {safe_split_uri(uri):uri for  uri in everything}

with open("mario.txt", "w") as f:
    f.write(str(lookup_table.keys()))

out_onto = Graph()
EX = Namespace("http://example.org/ontology#")
out_onto.bind("ex", EX)
out_onto.add((URIRef("http://example.org/ontology"), RDF.type, OWL.Ontology))


final_entities = set()
final_proprieties = set()

with open(home_path / "triples.txt", "r") as f:
    for line in f:

        s_label, p_label, o_label = line.strip().split("\t")
      
        s_uri = lookup_table.get(s_label)
        p_uri = lookup_table.get(p_label)
        o_uri = lookup_table.get(o_label)

        if s_uri and p_uri and o_uri:
            out_onto.add((s_uri, RDF.type, OWL.NamedIndividual))
            out_onto.add((o_uri, RDF.type, OWL.NamedIndividual))
            out_onto.add((p_uri, RDF.type, OWL.ObjectProperty))
            out_onto.add(((s_uri, p_uri, o_uri)))

            final_entities.add(s_uri)
            final_entities.add(o_uri)
            final_proprieties.add(p_uri)

final_classes = set()

for ind_uri in final_entities:

    classes = set(dbpedia_classes.objects(ind_uri, RDF.type))
    classes.discard(OWL.NamedIndividual)

    # If we find a individual that is of RDF.Type Owl.Class or does not have a class, we remove it 
    if (OWL.Class in classes) or (RDFS.Class in classes) or len(classes) == 0:
        print(ind_uri, classes)
        out_onto.remove((ind_uri, None, None))
        out_onto.remove((None, None, ind_uri))

    else:
        for owl_class in classes:
            out_onto.add((owl_class, RDF.type, OWL.Class))
            out_onto.add((owl_class, RDFS.subClassOf, OWL.Thing))
            out_onto.add((ind_uri, RDF.type, owl_class))

        final_classes.add(owl_class)


print("###")

for classes in final_classes:

    subclasses = set(dbpedia_full.objects(classes, RDFS.subClassOf))
    subclasses.discard(OWL.Thing)
    for subclass in subclasses:
        
        if subclass in final_classes:
            out_onto.add((classes, RDFS.subClassOf, subclass))
        else:
            out_onto.add((subclass, RDF.type, OWL.Class))
            out_onto.add((subclass, RDFS.subClassOf, OWL.Thing))
            out_onto.add((classes, RDFS.subClassOf, subclass))

    for elem in dbpedia_full.objects(classes, OWL.equivalentClass):
        if elem in final_classes:
            out_onto.remove((elem, None, None))
            out_onto.remove((None, None, elem))


for obj_uri in final_proprieties:
    domain_ = list(dbpedia_full.objects(obj_uri, RDFS.domain))
    range_ = list(dbpedia_full.objects(obj_uri, RDFS.range))
    
    for dom_uri in domain_:
        if dom_uri in final_classes:
            out_onto.add((obj_uri, RDFS.domain, dom_uri))

    for range_uri in range_:
        if range_uri in final_classes:
            out_onto.add((obj_uri, RDFS.range, range_uri))



out_onto.serialize("dbpedia_parsed.xml", format="xml")
