import numpy as np
from rdflib import URIRef, Namespace
from rdflib.namespace import OWL, RDF, RDFS, DC, DCTERMS, SKOS, VOID, FOAF, DOAP, XSD
from argparse import ArgumentParser
import re
from util import DAGNode, level_hierarchy
from scipy.sparse import coo_matrix, csc_matrix

nt_regex = r"<(.*)> <(.*)> <(.*)> \."
tsv_regex = r"<?(.*)>?\t<?(.*)>?\t<?(.*)>?[ \t]*"
#ignored_namespaces = [OWL, RDF, RDFS, DC, DCTERMS, SKOS, VOID, FOAF, DOAP, XSD]
ignored_namespaces = [OWL, RDF, RDFS, SKOS, VOID, XSD]

DBR = Namespace("http://dbpedia.org/resource/")
DBO = Namespace("http://dbpedia.org/ontology/")
DBP = Namespace("http://dbpedia.org/property/")

dctSubject = URIRef("http://purl.org/dc/terms/subject")


def filter_entity(e):
    for ns in ignored_namespaces:
        if str(e).startswith(str(ns)):
            return True
    return False


if __name__ == '__main__':

    parser = ArgumentParser("Loads Knowledge Graph from NT or TSV file into a tensor representation in numpy's "
                            "NPZ format, which can be used to train the models.")
    parser.add_argument("input", type=str, default=None, help="Path to input file (nt or tsv)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to output npz file")
    parser.add_argument("-nocat", "--no-categories", dest="nocat", action="store_true",
                        help="ignore dct:subject relations (categories in DBpedia)")
    parser.add_argument("-mt", "--materialize-types", dest="mattypes", action="store_true",
                        help="whether to materialize the types hierarchy")
    parser.add_argument("-mp", "--materialize-properties", dest="matprops", action="store_true",
                        help="whether to materialize the properties hierarchy")
    parser.add_argument("-mdr", "--materialize-domains-ranges", dest="matdomran", action="store_true",
                        help="whether to materialize the domain and range restrictions")
    parser.set_defaults(mattypes=False)
    parser.set_defaults(matprops=False)
    parser.set_defaults(matdomran=False)
    parser.set_defaults(nocat=False)


    args = parser.parse_args()

    input_path = args.input
    input_format = input_path[input_path.rindex(".") + 1:]

    assert input_format in ["nt", "tsv"]
    pattern = re.compile(nt_regex if input_format == "nt" else tsv_regex)

    dict_s = {}
    dict_p = {}
    dict_t = {}

    print("loading dictionaries")
    f = open(args.input, "rb")
    for line in f:
        if line:
            m = pattern.match(line)
            try:
                s, p, o = m.group(1), m.group(2), m.group(3)
                s, p, o = URIRef(s), URIRef(p), URIRef(o)

                if not args.nocat or p!= dctSubject:
                    if s.startswith(DBR):
                        if s not in dict_s:
                            dict_s[s] = len(dict_s)
                    if o.startswith(DBR):
                        if o not in dict_s:
                            dict_s[o] = len(dict_s)

                    if (p == RDF.type or p == "a") and not filter_entity(s) and not filter_entity(o):
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

    f = open(args.input, "rb")
    data_coo = [{"rows": [], "cols": [], "vals": []} for i in range(len(dict_p))]
    type_coo = {"rows": [], "cols": [], "vals": []}
    domains = {}
    ranges = {}
    type_dag = {}
    prop_dag = {}
    equiv_t = {}
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

                elif p == OWL.equivalentClass:
                    if s in dict_t and o in dict_t:
                        equiv_t[dict_t[s]] = dict_t[o]
                        equiv_t[dict_t[o]] = dict_t[s]
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

    print("load types hierarchy: total=%d matched=%d" % (type_total, type_matched))
    print("load relations hierarchy: total=%d matched=%d" % (prop_total, prop_matched))


    if len(equiv_t) > 0:
        print("adding class equivalences")
        typedata = typedata.tocsc()
        for t1,t2 in equiv_t.items():
            typedata[:,t1] = typedata[:,t1] = typedata[:,t1] + typedata[:,t2]
        typedata = typedata.tocsr()

    if args.matdomran:
        print("materializing domain and range restrictions")
        if domains is not None or ranges is not None:
            typedata_lil = typedata.tolil()
            for p,t in domains.items():
                for e in data[p].row:
                    if not typedata[e,t]:
                        typedata_lil[e,t] = 1
            for p,t in ranges.items():
                for e in data[p].col:
                    if not typedata[e,t]:
                        typedata_lil[e,t] = 1
            typedata = typedata_lil.tocsr()

    if args.mattypes:
        print("materializing types hierarchy")
        if type_dag:
            n1 = typedata.nnz
            typedata = typedata.tocsc()
            rows = [list(typedata[:,i].indices) for i in range(typedata.shape[1])]
            th_levels = level_hierarchy(type_dag)

            for nodes in reversed(th_levels[1:]):
                for n in nodes:
                    for p in n.parents:
                        rows[p] = rows[p] + rows[n.node_id]


            col = sum([[i]*len(row_i) for i,row_i in enumerate(rows)], [])
            row = sum(rows,[])
            val = [True]*len(col)

            typedata = coo_matrix((val,(row,col)),shape=typedata.shape, dtype=typedata.dtype)
            typedata = typedata.tocsr()
            n2 = typedata.nnz
            print("%d type assertions add by reasoning subClassOf relations" % (n2 - n1))

    if args.matprops:
        print("materializing properties hierarchy")
        if prop_dag:
            n1 = sum([A.nnz for A in data])
            ph_levels = level_hierarchy(prop_dag)
            for nodes in reversed(ph_levels[1:]):
                for n in nodes:
                    for p in n.parents:
                        data[p] = data[p] + data[n.node_id]
            n2 = sum([A.nnz for A in data])
            print("%d relation assertions added by reasoning subPropertyOf relations" % (n2 - n1))

    if args.output is None:
        args.output = args.input.replace("." + input_format, ".npz")

    np.savez(args.output,
             data=data,
             types=typedata,
             entities_dict=dict_s,
             relations_dict=dict_p,
             types_dict=dict_t,
             type_hierarchy=type_dag,
             prop_hierarchy=prop_dag,
             domains=domains,
             ranges=ranges)
