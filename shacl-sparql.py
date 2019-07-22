import pickle
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np

from util import short_str


def xor(x, y):
    return (x and not y) or (not x and y)


def prune_redundancies(dt, i):
    left, right = True, True
    if dt.children_left[i] > -1:
        left = prune_redundancies(dt, dt.children_left[i])
    if dt.children_right[i] > -1:
        right = prune_redundancies(dt, dt.children_right[i])

    if left and right:
        dt.children_left[i] = -1
        dt.children_right[i] = -1

    return not xor(dt.children_left[i] > -1, dt.children_right[i] > -1)


def prune_redundancies_cond(dt, minsup, condition):
    changed = 1
    while changed:
        changed = 0
        for i in range(dt.node_count):
            if (condition(dt.value[i, 0]) and sum(dt.value[i, 0]) >= minsup):
                dt.children_left[i] = -1
                dt.children_right[i] = -1

    return dt


def prune_dt(dt, minsup, condition):
    changed = 1
    while changed:
        changed = 0
        for i in range(dt.node_count):
            if (dt.children_left[i], dt.children_right[i]) == (-1, -1) and \
                    not (condition(dt.value[i, 0]) and sum(dt.value[i, 0]) >= minsup):
                if i in dt.children_left:
                    idx = list(dt.children_left).index(i)
                    changed += int(dt.children_left[idx] != -1)
                    dt.children_left[idx] = -1
                if i in dt.children_right:
                    idx = list(dt.children_right).index(i)
                    changed += int(dt.children_right[idx] != -1)
                    dt.children_right[idx] = -1
    return dt


def negate(s):
    if s.startswith("! ") :
        return s[2:]
    else:
        return "! " + s + " "


def create_rules(m, r, rels_dict, types_dict, minsup, cond):
    if m is None or m.models[r] is None:
        return ""

    dt = deepcopy(m.models[r].tree_)
    dt = prune_dt(dt, minsup, cond)
    dt = prune_redundancies_cond(dt, minsup, cond)


    n_rel = len(rels_dict)

    def get_condition_str(i, neg=False):
        f = dt.feature[i]
        if f < len(m.selected_paths[r]):
            path = "/".join(["<" + rels_dict[x] + ">" if x < n_rel else "^<" + rels_dict[x - n_rel] + ">" for x in
                 m.selected_paths[r][f]])
            s = "{ $this " + path + " ?o } "

        elif f < (len(m.selected_paths[r]) + len(m.selected_s_types[r])):
            f -= len(m.selected_paths[r])
            s = "{ $this a <" + types_dict[m.selected_s_types[r][f]] + "> } "

        elif f < (len(m.selected_paths[r]) + len(m.selected_s_types[r]) + len(m.selected_o_types[r])):
            f -= len(m.selected_paths[r]) + len(m.selected_s_types[r])
            s = "{ ?o a <" + types_dict[m.selected_o_types[r][f]] + "> } "

        elif m.so_iorels_feat:
            f1 = m.selected_out_s_feats[r]
            f2 = m.selected_out_o_feats[r]
            f3 = m.selected_in_s_feats[r]
            f4 = m.selected_in_o_feats[r]
            offset = len(m.selected_paths[r]) + len(m.selected_s_types[r]) + len(m.selected_o_types[r])
            path = tuple([])

            if f < offset + len(f1):
                path = f1[f - offset]
                e1,e2 = "$this", "?X"
            elif f < offset + len(f1) + len(f2):
                path = f2[f - offset - len(f1)]
                e1, e2 = "?o", "?X"
            elif f < offset + len(f1) + len(f2) + len(f3):
                path = f3[f - offset - len(f1) - len(f2)]
                e1, e2 = "?X", "$this"
            elif f < offset + len(f1) + len(f2) + len(f3) + len(f4):
                path = f4[f - offset - len(f1) - len(f2) - len(f3)]
                e1, e2 = "?X", "?o"
            path_str = "/".join(["<" + rels_dict[x] + ">" if x < n_rel else "^<" + rels_dict[x - n_rel] + ">" for x in
                      path])
            s = "{ " + e1 + " " + path_str + " " + e2 + " } "
        s = "EXISTS " + s
        if neg:
            s = negate(s)
        return s


    # Separate root node to avoid negating main expression
    def create_conditions_root(i):
        s1, s2 = "", ""
        if dt.children_left[i] > 0:
            s1 = get_condition_str(i, neg=True)
            idx = dt.children_left[i]
            if dt.children_left[idx] > 0 or dt.children_right[idx] > 0:
                s11 = create_conditions_root(idx)
                s1 = "( " + s1 + "  &&  " + s11 + " )"
        if dt.children_right[i] > 0:
            s2 = get_condition_str(i, neg=False)
            idx = dt.children_right[i]
            if dt.children_left[idx] > 0 or dt.children_right[idx] > 0:
                s22 = create_conditions_root(idx,)
                s2 = "( " + s2 + "  &&  " + s22 + " )"
        if dt.children_left[i] > 0 and dt.children_right[i] > 0:
            s = "( " + s1 + "  ||  " + s2 + " )"
        else:
            s = s1 + s2

        return s

    r_name = short_str(rels_dict[r])

    #conditions = create_conditions(0)
    conditions = create_conditions_root(0)

    if len(conditions) == 0:
        return ""
    else:
        ttl = ""
        ttl += ":" + r_name + "Shape a sh:NodeShape ;\n"
        ttl += "sh:targetSubjectsOf <" + rels_dict[r] + "> ;\n"
        ttl += "sh:sparql [\n"
        ttl += "  a sh:SPARQLConstraint;\n"
        ttl += "  sh:select \"\"\" \n"
        ttl += "SELECT $this ?o WHERE { $this <" + rels_dict[r] + "> ?o . \n"
        ttl += "FILTER(("+conditions+")) } \n"
        ttl += "  \"\"\" ;\n "
        ttl += "] . "
        return ttl


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("model", type=str, default=None, help="Path to PaTyBRED model pickle file")
    parser.add_argument("data", type=str, default=None, help="Path to data npz file")
    parser.add_argument("-c", "--conf", type=float, default=1, help="Error confidence")
    parser.add_argument("-ms", "--minsup", type=int, default=5, help="Minimum leaf node support")
    parser.add_argument("-r", "--rels", type=int, nargs="+", default=None, help="relations list (if None do all)")

    parser.add_argument("-xp", "--exclude-pure", dest="no_pure", action="store_true", help="Exclude pure error nodes")
    parser.set_defaults(no_pure=False)

    args = parser.parse_args()

    assert not (args.no_pure and args.conf == 1)

    if args.no_pure:
        condition = lambda x, conf=args.conf: x[1] > 0 and float(x[0]) / sum(x) >= conf
    else:
        condition = lambda x, conf=args.conf: (float(x[0]) / sum(x)) >= conf

    # print("#loading patybred model")
    m = pickle.load(open(args.model, "rb"))

    # print("#loading dictionaries")
    d = np.load(args.data, allow_pickle=True)
    rels_dict = {k: v for v, k in d["relations_dict"].item().items()}
    type_dict = {k: v for v, k in d["types_dict"].item().items()}

    minsup = 50

    n_rels = len(rels_dict)

    ttl = ""
    ttl += "@prefix : <http://patybred.shacl/> . \n"
    ttl += "@prefix sh: <http://www.w3.org/ns/shacl#> . \n"

    if args.rels is None:
        args.rels = range(n_rels)

    for r in args.rels:
        ttl_r = create_rules(m, r, rels_dict, type_dict, minsup, cond=condition)
        ttl += ttl_r + "\n\n" if len(ttl_r) > 0 else ""

    print(ttl)
