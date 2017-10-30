import numpy as np
from rdflib import URIRef
from rdflib.namespace import OWL, RDF, RDFS, DC, DCTERMS, SKOS, VOID, FOAF, DOAP, XSD
from argparse import ArgumentParser
import re
from util import DAGNode, level_hierarchy
from scipy.sparse import coo_matrix

nt_regex = r"<(.*)> <(.*)> <(.*)> \."
tsv_regex = r"<?(.*)>?\t<?(.*)>?\t<?(.*)>?[ \t]*"
#ignored_namespaces = [OWL, RDF, RDFS, DC, DCTERMS, SKOS, VOID, FOAF, DOAP, XSD]
ignored_namespaces = [OWL, RDF, RDFS, SKOS, VOID, XSD]


def filter_entity(e):
    for ns in ignored_namespaces:
        if str(e).startswith(str(ns)):
            return True
    return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="Path to input file (nt or tsv)")

    args = parser.parse_args()

    input_path = args.input
    input_format = input_path[input_path.rindex(".") + 1:]

    assert input_format in ["nt", "tsv"]
    pattern = re.compile(nt_regex if input_format == "nt" else tsv_regex)

    dict_s = {}
    dict_p = {}
    dict_t = {}
    equiv_t = {}

    print("loading dictionaries")
    f = file(args.input, "rb")
    for line in f:
        if line:
            m = pattern.match(line)
            try:
                s, p, o = m.group(1), m.group(2), m.group(3)
                s, p, o = URIRef(s), URIRef(p), URIRef(o)

                if p == RDF.type and not filter_entity(s) and not filter_entity(o):
                    if s not in dict_s:
                        dict_s[s] = len(dict_s)
                    if o not in dict_t:
                        dict_t[o] = len(dict_t)
                elif p == RDFS.subClassOf:
                    if s not in dict_t:
                        dict_t[s] = len(dict_t)
                    if o not in dict_t:
                        dict_t[o] = len(dict_t)
                elif p == RDFS.subPropertyOf:
                    if s not in dict_p:
                        dict_p[s] = len(dict_p)
                    if o not in dict_p:
                        dict_p[o] = len(dict_p)
                elif not filter_entity(p):
                    if p not in dict_p:
                        dict_p[p] = len(dict_p)
            except:
                continue
    print("%d entities, %d types, %d properties" % (len(dict_s), len(dict_t), len(dict_p)))

    f = file(args.input, "rb")
    data_coo = [{"rows": [], "cols": [], "vals": []} for i in range(len(dict_p))]
    type_coo = {"rows": [], "cols": [], "vals": []}
    domains = {}
    ranges = {}
    type_dag = {}
    prop_dag = {}
    print("loading data")
    for line in f:
        if line:
            m = pattern.match(line)
            try:
                s, p, o = m.group(1), m.group(2), m.group(3)
                s, p, o = URIRef(s), URIRef(p), URIRef(o)

                if p == RDF.type:
                    if s in dict_s and o in dict_t:
                        type_coo["rows"].append(dict_s[s])
                        type_coo["cols"].append(dict_t[o])
                        type_coo["vals"].append(True)

                elif p in dict_p:
                    if s in dict_s and o in dict_s:
                        p_data_coo = data_coo[dict_p[p]]
                        p_data_coo["rows"].append(dict_s[s])
                        p_data_coo["cols"].append(dict_s[o])
                        p_data_coo["vals"].append(True)

                elif p == RDFS.subClassOf:
                    if s in dict_t and o in dict_t:
                        s_id = dict_t[s]
                        o_id = dict_t[o]
                        if s_id != o_id:
                            if o_id not in type_dag:
                                type_dag[o_id] = DAGNode(o_id, o, parents=[], children=[])
                            if s_id not in type_dag:
                                type_dag[s_id] = DAGNode(s_id, s, parents=[], children=[])

                            type_dag[s_id].parents.append(type_dag[o_id])
                            type_dag[o_id].children.append(type_dag[s_id])

                elif p == RDFS.subPropertyOf:
                    if s in dict_p and o in dict_p:
                        s_id = dict_p[s]
                        o_id = dict_p[o]
                        if s_id != o_id:
                            if o_id not in prop_dag:
                                prop_dag[o_id] = DAGNode(o_id, o, parents=[], children=[])
                            if s_id not in prop_dag:
                                prop_dag[s_id] = DAGNode(s_id, s, parents=[], children=[])

                            prop_dag[s_id].parents.append(prop_dag[o_id])
                            prop_dag[o_id].children.append(prop_dag[s_id])

                elif p == RDFS.domain:
                    if s in dict_p and o in dict_t:
                        domains[dict_p[s]] = dict_t[o]

                elif p == RDFS.range:
                    if s in dict_p and o in dict_t:
                        ranges[dict_p[s]] = dict_t[o]
            except:
                continue

    data = [coo_matrix((p["vals"], (p["rows"], p["cols"])), shape=(len(dict_s), len(dict_s)), dtype=bool) for p in
            data_coo]
    typedata = coo_matrix((type_coo["vals"], (type_coo["rows"], type_coo["cols"])), shape=(len(dict_s), len(dict_t)),
                          dtype=bool)

    # change from objects to indices to avoid "maximum recursion depth exceeded" when pickling
    for i, n in type_dag.items():
        n.children = [c.node_id for c in n.children]
        n.parents = [p.node_id for p in n.parents]
    for i, n in prop_dag.items():
        n.children = [c.node_id for c in n.children]
        n.parents = [p.node_id for p in n.parents]

    type_total = len(dict_t) if dict_t else 0
    type_matched = len(type_dag) if type_dag else 0
    prop_total = len(dict_p) if dict_p else 0
    prop_matched = len(prop_dag) if prop_dag else 0

    print "load types hierarchy: total=%d matched=%d" % (type_total, type_matched)
    print "load relations hierarchy: total=%d matched=%d" % (prop_total, prop_matched)

    print "materializing types hierarchy"

    if type_dag:
        n1 = typedata.nnz
        typedata = typedata.tocsc()
        th_levels = level_hierarchy(type_dag)
        for nodes in reversed(th_levels[1:]):
            for n in nodes:
                for p in n.parents:
                    typedata[:, p] = typedata[:, p] + typedata[:, n.node_id]
        typedata = typedata.tocsr()
        n2 = typedata.nnz
        print "%d type assertions add by reasoning subClassOf relations" % (n2 - n1)

    print "materializing properties hierarchy"
    if prop_dag:
        n1 = sum([A.nnz for A in data])
        ph_levels = level_hierarchy(prop_dag)
        for nodes in reversed(ph_levels[1:]):
            for n in nodes:
                for p in n.parents:
                    data[p] = data[p] + data[n.node_id]
        n2 = sum([A.nnz for A in data])
        print "%d relation assertions added by reasoning subPropertyOf relations" % (n2 - n1)


    np.savez(args.input.replace("." + input_format, ".npz"),
             data=data,
             types=typedata,
             entities_dict=dict_s,
             relations_dict=dict_p,
             types_dict=dict_t,
             type_hierarchy=type_dag,
             prop_hierarchy=prop_dag,
             domains=domains,
             ranges=ranges)
