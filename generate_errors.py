import sys

from argparse import ArgumentParser
from util import loadGraphNpz, loadTypesNpz, load_entities_dict, load_types_dict, \
    load_domains, load_ranges, load_relations_dict, to_triples, load_type_hierarchy, load_prop_hierarchy
import numpy as np
from skge.sample import LCWASampler, CorruptedSampler, RandomSampler
from random import randint
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict as ddict
from tqdm import tqdm
from copy import deepcopy


def generate_wrong_fact(X, types=None, domains=None, ranges=None, kind=1, p_error=0.001):

    assert kind == 1 or (kind == 2 and types is not None)

    triples = to_triples(X, order="sop", dtype="list")

    types_csc = types.tocsc(copy=True) if types is not None and types.nnz else None

    n_entities = X[0].shape[0]
    n_facts = sum([xi.nnz for xi in X])
    n_wrong_facts = int(n_facts * p_error)

    n_types = types.shape[1]
    omnipresent_types = [i for i in range(n_types) if types_csc is not None and types_csc[:,i].nnz == n_entities]

    print("%d erroneous facts of kind %d to be generated" % (n_wrong_facts, kind))

    sample = np.random.choice(len(triples), n_wrong_facts, replace=False)
    to_be_corrupted = [triples[i] for i in sample]

    if kind > 0:
        print("Creating type groups")
        pbar = tqdm(total=types.shape[0])
        types_entitities = ddict(lambda: set())
        for i, x in enumerate(types):
            types_entitities[tuple(x.indices)].add(i)
            pbar.update(1)
        pbar.close()

    errors_list = []
    print("Creating erroneous facts")
    pbar = tqdm(total=len(to_be_corrupted))
    for t in to_be_corrupted:
        corrupted = list(t)

        # randomly choose between corrupt subject or object
        t_i = randint(0, 1)
        if kind > 0:
            while True:
                t_i = randint(0, 1)
                e = t[t_i]
                e_types = tuple(types[e].indices)
                same_types = types_entitities[e_types]
                if len(same_types) > 1 or kind != 1:
                    break

        if kind and len(same_types) == n_entities and kind > 1:
            same_types = set()


        if kind == 1:
            while True:
                corrupted[t_i] = randint(0, n_entities - 1)
                if corrupted not in triples and corrupted not in errors_list:
                    break

        elif kind == 2:
            candidates = [i for i in same_types if i != e]
            while True:
                corrupted[t_i] = candidates[randint(0, len(candidates) - 1)]
                if corrupted not in triples and corrupted not in errors_list:
                    break

        else:
            raise("Error kind %d is not supported. Please choose from [1,2]\n%s" % kind)

        errors_list.append(tuple(corrupted))

        pbar.update(1)

    pbar.close()

    return errors_list


def update_data(X, errors):
    cols = [[] for p in range(len(X))]
    rows = [[] for p in range(len(X))]
    data = [[] for p in range(len(X))]
    for s, o, p in errors:
        rows[p].append(s)
        cols[p].append(o)
        data[p].append(True)

    for p in range(len(X)):
        X[p] = coo_matrix((list(X[p].data) + data[p], (list(X[p].row) + rows[p], list(X[p].col) + cols[p])),
                          shape=X[p].shape)

    return X


if __name__ == '__main__':
    parser = ArgumentParser(
        "Generates error detection data in knowledge graphs by randomly corrupting triples in order to generate wrong facts. "
        "These wrong facts can be of three kinds:"
        " 1 - Randomly corrupted triple"
        " 2 - Same type as the original entity")

    parser.add_argument("input", type=str, default=None, help="Path of the input npz kb file")
    parser.add_argument("-pe", "--p-error", type=float, default=0.01, help="Proportion of errors to be generated")
    parser.add_argument("-ek", "--error-kind", type=int, default=1, help="Kind of errors to be generated [1,2]")

    args = parser.parse_args()

    output_path = args.input.replace(".npz", "-errdet-ek%d-p%f.npz" % (args.error_kind, args.p_error))

    d = np.load(args.input)
    X = d["data"]
    types = d["types"].item()
    domains = d["domains"].item()
    ranges = d["ranges"].item()
    entities_dict = d["entities_dict"].item()
    relations_dict = d["relations_dict"].item()
    types_dict = d["types_dict"].item()
    type_hier = None
    prop_hier = None

    entities_dict = {k: v for v,k in entities_dict.items()}
    relations_dict = {k: v for v, k in relations_dict.items()}


    # if not isinstance(types, csr_matrix):
    #    if not isinstance(types, coo_matrix):
    types = coo_matrix(types)
    types = types.tocsr()

    errors_list = generate_wrong_fact(X, types, domains, ranges, args.error_kind, args.p_error)

    X = update_data(X, errors_list)

    # Change node objects to ids to avoid maximum recursion depth
    if type_hier is not None:
        for i, n in type_hier.items():
            n.children = [c.node_id for c in n.children]
            n.parents = [p.node_id for p in n.parents]
    if prop_hier is not None:
        for i, n in prop_hier.items():
            n.children = [c.node_id for c in n.children]
            n.parents = [p.node_id for p in n.parents]

    np.savez(output_path, data=X,
             types=types,
             domains=domains,
             ranges=ranges,
             entities_dict=entities_dict,
             types_dict=types_dict,
             relations_dict=relations_dict,
             type_hierarchy=type_hier,
             prop_hierarchy=prop_hier,
             errors=errors_list)
