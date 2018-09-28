from sdvalidate import SDValidate
from argparse import ArgumentParser
from patybred import PaTyBRED
from embeddings import SKGEWrapper, ProjE
from util import to_triples, short_str
import numpy as np
from datetime import datetime
from entityasm import EntityASM, EntityDisamb
import pickle
from errordetector import ErrorDetector
import scipy.sparse as sp



def correct(triples, scores, ed, asm, tp=None, tpdata=None, disamb=None, p=0.01, min_score=0.75, min_score_gain=1.5, max_dist=2, domains=None,
            ranges=None):
    global rels_dict
    global ents_dict
    n = int(len(triples) * p)
    types = ed.types

    if tp is not None and tpdata is not None:
        inv_ents_dict = {k:v for v,k in ents_dict.items()}
        tpd_ents_dict = tpdata["entities_dict"].item()
        if not isinstance(tpd_ents_dict.keys()[0], int):
            tpd_ents_dict = {k: v for v, k in tpd_ents_dict.items()}

        tpd_ents_dict = {inv_ents_dict[ent]:i for i,ent in tpd_ents_dict.items()}
        tpfeats = sp.coo_matrix(tpdata["feats"].item()).tocsr()

    for i, (s, o, p) in enumerate(triples[:n]):

        score = scores[i] if scores is not None else ed.predict_proba([(s, o, p)])[0]

        s_ent = ents_dict[s]
        o_ent = ents_dict[o]

        # Checks if type prediction can improve the scores of the triple
        if types is not None and tp is not None and tpdata is not None:
            relevant_s_types = ed.selected_s_types[p]
            relevant_o_types = ed.selected_o_types[p]

            if relevant_s_types:
                s_types = tp.predict(tpfeats[tpd_ents_dict[s]])[0]
                #print(short_str(ents_dict[s]), s_types)
                if list(types[s, relevant_s_types]) != list(s_types[relevant_s_types]):
                    ed.types[s, relevant_s_types] = s_types[relevant_s_types]
                    new_score = ed.predict_proba([(s, o, p)])[0]
                    ed.types[s, relevant_s_types] = types[s, relevant_s_types]
                    if new_score / score >= min_score_gain and new_score > min_score:
                        old_types = [i for i in sp.csr_matrix(types[s]).indices]
                        new_types = [i for i in sp.csr_matrix(s_types).indices]
                        triple_str = str([short_str(rels_dict[p]),short_str(ents_dict[s]),short_str(ents_dict[o])])
                        print("Triple = %s, Entity %s [old types = %s] [new types = %s]"%(triple_str, short_str(ents_dict[s]), old_types, new_types))
                        continue

            if relevant_o_types:
                o_types = tp.predict(tpfeats[tpd_ents_dict[o]])[0]
                #print(short_str(ents_dict[o]), o_types)
                if list(types[o, relevant_o_types]) != list(o_types[relevant_o_types]):
                    ed.types[o, relevant_o_types] = o_types[relevant_o_types]
                    new_score = ed.predict_proba([(s, o, p)])[0]
                    ed.types[o, relevant_o_types] = types[o, relevant_o_types]
                    if new_score / score >= min_score_gain and new_score > min_score:
                        old_types = [i for i in sp.csr_matrix(types[o]).indices]
                        new_types = [i for i in sp.csr_matrix(o_types).indices]
                        triple_str = str([short_str(rels_dict[p]), short_str(ents_dict[s]), short_str(ents_dict[o])])
                        print("Triple = %s, Entity %s [old types = %s] [new types = %s]"%(triple_str, short_str(ents_dict[o]), old_types, new_types))
                        continue



        # In DBpedia confusions normally occur on the object
        s_candidates = []
        #s_candidates = asm.get_similar_entities(s_ent, id=True, silent=True, match_all_words=True)
        o_candidates = asm.get_similar_entities(o_ent, id=True, silent=True, match_all_words=True)

        if types is not None:
            if domains is not None:
                s_candidates = [(s_cand,dist) for s_cand, (freq, dist, ent_ids, w) in s_candidates if types[s_cand,domains[p]] and dist <= max_dist and (s_cand,o,p) not in triples]

            if ranges is not None:
                o_candidates = [(o_cand,dist) for o_cand, (freq, dist, ent_ids, w) in o_candidates if types[o_cand, ranges[p]] and dist <= max_dist and (s,o_cand,p) not in triples]

        if disamb is not None:
            s_candidates += [(s_cand,1) for s_cand in disamb.suggestions(s) if types[s_cand, domains[p]] and (s_cand,o,p) not in triples]
            o_candidates += [(o_cand,1) for o_cand in disamb.suggestions(o) if types[o_cand, ranges[p]] and (s,o_cand,p) not in triples]

        fixed_triple = None
        max_score = score
        min_dist = float("inf")

        if types is None or types[s].nnz > 1:
            for s_cand, dist in s_candidates:
                if s_cand != s:
                    cand_score = ed.predict_proba([(s_cand, o, p)])[0]
                    if cand_score / score >= min_score_gain and cand_score > min_score and cand_score >= max_score:
                        if cand_score > max_score or dist < min_dist:
                            max_score = cand_score
                            min_dist = dist
                            fixed_triple = (s_cand, o, p)

        if types is None or types[o].nnz > 1:
            for o_cand, dist in o_candidates:
                if o_cand != o:
                    cand_score = ed.predict_proba([(s, o_cand, p)])[0]
                    if cand_score / score >= min_score_gain and cand_score > min_score and cand_score >= max_score:
                        if cand_score > max_score or dist < min_dist:
                            max_score = cand_score
                            min_dist = dist
                            fixed_triple = (s, o_cand, p)

        if fixed_triple is not None:
            print(
                "old triple [%f]: %s(%s , %s)" % (
                score, short_str(rels_dict[p]), short_str(ents_dict[s]), short_str(ents_dict[o])))
            s, o, p = fixed_triple
            print(
                "new triple [%f]: %s(%s , %s)" % (
                max_score, short_str(rels_dict[p]), short_str(ents_dict[s]), short_str(ents_dict[o])))


if __name__ == '__main__':
    parser = ArgumentParser(description="Type prediction evalutation with cross-validation")
    parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
    parser.add_argument("-disamb", "--disambiguation", type=str, default=None,
                        help="path to pkl of disambiguation object")
    parser.add_argument("-tsc", "--triple-scores", type=str, default=None,
                        help="path to triples scores file")
    parser.add_argument("-e", "--embeddings", type=str, default=None, help="path to dataset with hole embeddings")
    parser.add_argument("-asm", "--approx-sim", type=str, default=None, help="path to ASM object")
    parser.add_argument("-tp", "--type-predictor", type=str, default=None, help="path to type prediction pkl file")
    parser.add_argument("-tpd", "--type-prediction-data", type=str, default=None, help="path to type prediction data npz file")
    parser.add_argument("-lp", "--load-path", type=str, default=None, help="path to load trained model")
    parser.add_argument("-sp", "--save-path", type=str, default=None, help="path to save trained model")
    parser.add_argument("-fs", "--feature-selection", type=str, default="chi2", help="feature selection method")
    parser.add_argument("-mf", "--max-feats", type=int, default=10, help="number of features to be selected")
    parser.add_argument("-negw", "--neg-weight", type=float, default=1, help="negative examples selection weight")
    parser.add_argument("-nneg", "--n-negatives", type=int, default=1, help="negative examples selection weight")
    parser.add_argument("-mpl", "--max-path-length", type=int, default=2, help="maximum path length (PRA)")
    parser.add_argument("-mppl", "--max-paths-per-level", type=int, default=99999999,
                        help="maximum number of paths per level (PRA), prune the potentially large search space with heuristic based on the number of intersecting object and subject of a path link")
    parser.add_argument("-minsup", "--minimum-support", type=float, default=0.001, help="minimum path support (PRA)")
    parser.add_argument("-d", "--dimensions", type=int, default=100, help="number of embeddings dimensions")
    parser.add_argument("-ckp", "--checkpoint-freq", type=int, default=0,
                        help="frequency with which the model is saved to a checkpoint")
    parser.add_argument("-ne", "--n-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-mts", "--max-ts", type=int, default=2500, help="maximum local training set size")
    parser.add_argument("-mfs", "--max-fs", type=int, default=500, help="maximum feature selection data size")
    parser.add_argument("-od", "--outlier-detection-method", type=str, default="if", help="outlier detection method")
    parser.add_argument("-clf", "--classifier", type=str, default="rf", help="classification method")
    parser.add_argument("-psm", "--path-selection-mode", type=str, default="m2", help="path selection mode")
    parser.add_argument("-m", "--method", type=str, default="sdv",
                        help="multilabel classifier to be used for type prediction")
    parser.add_argument("-msg", "--min-score-gain", type=float, default=1.5,
                        help="minimum score gain to correct triple")

    parser.add_argument("-sok", "--convert-to-sok", dest="sok", action="store_true",
                        help="convert csr_matrix to sok_matrix make cell access faster")
    parser.add_argument("-mc", "--mem-cache", dest="mem_cache", action="store_true",
                        help="use cache and evict to disk to reduce memory usage")
    parser.add_argument("-pp", "--plot", dest="plot", action="store_true", help="plot PRAUC")
    parser.add_argument("-ut", "--use-types", dest="use_types", action="store_true",
                        help="whether to use type assertions for learning embeddings")
    parser.set_defaults(mem_cache=False)
    parser.set_defaults(sok=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(use_types=False)

    args = parser.parse_args()

    print(args)

    hist_path = args.input.replace(".npz", "-" + args.method + "-scores-dist.png")
    scores_path = args.input.replace(".npz", "-" + args.method + "-scores.pkl")

    d = np.load(args.input)
    X = None
    types = d["types"].item()
    domains = d["domains"].item()
    ranges = d["domains"].item()
    ents_dict = d["entities_dict"](args.input)
    rels_dict = d["relations_dict"](args.input)
    type_hierarchy = None


    if not isinstance(ents_dict.keys()[0], int):
        ents_dict = {k: v for v, k in ents_dict.items()}
    if not isinstance(rels_dict.keys()[0], int):
        rels_dict = {k: v for v, k in rels_dict.items()}


    if args.load_path is None:
        X = d["data"]
        n_relations = len(X)
        n_entities = X[0].shape[0]
        n_types = types.shape[1] if types is not None else 0

        print("%d entities, %d relations, %d types" % (n_entities, n_relations, n_types))

        if args.method == "sdv":
            if args.load_path is None:
                ed = SDValidate()
            else:
                ed = SDValidate.load_model(args.load_path)
        if args.method == "pabred":
            ed = PaTyBRED(max_depth=args.max_path_length, clf_name=args.classifier, so_type_feat=False,
                          n_neg=args.n_negatives, lfs=args.feature_selection, max_feats=args.max_feats,
                          min_sup=args.minimum_support, max_paths_per_level=args.max_paths_per_level,
                          path_selection_mode=args.path_selection_mode, reduce_mem_usage=args.mem_cache,
                          convert_to_sok=args.sok, max_pos_train=args.max_ts, max_fs_data_size=args.max_fs)

        if args.method == "tybred":
            ed = PaTyBRED(max_depth=0, max_paths_per_level=0, clf_name=args.classifier, so_type_feat=True,
                          n_neg=args.n_negatives, lfs=args.feature_selection, max_feats=args.max_feats,
                          min_sup=args.minimum_support,
                          path_selection_mode=args.path_selection_mode, reduce_mem_usage=args.mem_cache,
                          convert_to_sok=args.sok, max_pos_train=args.max_ts, max_fs_data_size=args.max_fs)

        if args.method == "patybred":
            ed = PaTyBRED(max_depth=args.max_path_length, clf_name=args.classifier, so_type_feat=True,
                          n_neg=args.n_negatives, lfs=args.feature_selection, max_feats=args.max_feats,
                          min_sup=args.minimum_support, max_paths_per_level=args.max_paths_per_level,
                          path_selection_mode=args.path_selection_mode, reduce_mem_usage=args.mem_cache,
                          convert_to_sok=args.sok, max_pos_train=args.max_ts, max_fs_data_size=args.max_fs)

        if args.method in ["transe", "hole", "rescal"]:
            ed = SKGEWrapper(n_dim=args.dimensions, model=args.method, negative_examples=args.n_negatives)
        if args.method == "proje":
            ed = ProjE(args.dimensions, n_relations, n_entities, max_iter=args.n_epochs, add_types=args.use_types,
                       save_per=args.checkpoint_freq, neg_weight=args.neg_weight)

        t1 = datetime.now()
        ed.learn_model(X, types, type_hierarchy, domains, ranges)
        t2 = datetime.now()
        print("Training time = %f s" % (t2 - t1).total_seconds())
        if args.save_path is not None:
            ed.save_model(args.save_path)
    else:
        ed = ErrorDetector.load_model(args.load_path)
        print("Model loaded from %s" % args.load_path)

    if args.triple_scores is None:
        if X is None:
            X = d["data"]
        triples = to_triples(X, order="sop", dtype="list")
        t1 = datetime.now()
        scores = ed.predict_proba(triples)
        scores = scores.flatten()
        t2 = datetime.now()
        print("Prediction time = %f" % (t2 - t1).total_seconds())

        sorted_triples = [triples[i] for i in np.argsort(scores)]
        scores.sort()
    else:
        score_triples = pickle.load(file(args.triple_scores,"rb"))
        scores = [score for rank, score, triple in score_triples]
        sorted_triples = [triple for rank, score, triple in score_triples]

    print("Creating approximate string matching for entities")
    if args.approx_sim is None:
        asm = EntityASM(max_edit_distance=1, k=100, verbose=False)
        asm.create_dictionary(ents_dict)
        #asm.save("asm-dbpedia-2016.pkl")
    else:
        asm = EntityASM.load(args.approx_sim)

    disamb = None
    if args.disambiguation is not None:
        disamb = EntityDisamb(args.disambiguation, ents_dict)

    tp = None
    tpd = None
    if args.type_predictor is not None and args.type_prediction_data is not None:
        tp = pickle.load(file(args.type_predictor,"rb"))
        tpdata = np.load(args.type_prediction_data)

    correct(sorted_triples, scores, ed, asm, tp, tpdata, disamb, min_score_gain=args.min_score_gain)
