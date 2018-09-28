import sys
from util import short_str, to_triples, plot_histogram
from sdvalidate import SDValidate
import pickle
from argparse import ArgumentParser
from patybred import PaTyBRED
from bblbred import BblBRED
import numpy as np
from embeddings import ProjE, SKGEWrapper
from scipy.stats import rankdata
from datetime import datetime
import scipy.sparse as sp

if __name__ == '__main__':
    parser = ArgumentParser(description="Learns a KG model and ranks its facts based on confidence scores")
    parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
    parser.add_argument("-o", "--output", type=str, default=None, help="path output ranking file")
    parser.add_argument("-lp", "--load-path", type=str, default=None, help="path to load trained model")
    parser.add_argument("-sp", "--save-path", type=str, default=None, help="path to save trained model")
    parser.add_argument("-fs", "--feature-selection", type=str, default="chi2", help="feature selection method")
    parser.add_argument("-mf", "--max-feats", type=int, default=10, help="number of features to be selected")
    parser.add_argument("-negw", "--neg-weight", type=float, default=1, help="negative examples selection weight")
    parser.add_argument("-nneg", "--n-negatives", type=int, default=1, help="negative examples selection weight")
    parser.add_argument("-mpl", "--max-path-length", type=int, default=2, help="maximum path length (PRA)")
    parser.add_argument("-mppl", "--max-paths-per-level", type=int, default=sys.maxsize,
                        help="maximum number of paths per level (PRA), prune the potentially large search space with heuristic based on the number of intersecting object and subject of a path link")
    parser.add_argument("-minsup", "--minimum-support", type=float, default=0.001, help="minimum path support (PRA)")
    parser.add_argument("-d", "--dimensions", type=int, default=100, help="number of embeddings dimensions")
    parser.add_argument("-ckp", "--checkpoint-freq", type=int, default=0,
                        help="frequency with which the model is saved to a checkpoint")
    parser.add_argument("-ne", "--n-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-od", "--outlier-detection-method", type=str, default="if", help="outlier detection method")
    parser.add_argument("-clf", "--classifier", type=str, default="rf", help="classification method")
    parser.add_argument("-psm", "--path-selection-mode", type=str, default="m2", help="path selection mode")
    parser.add_argument("-mfs", "--max-fs", type=int, default=500, help="maximum feature selection data size")
    parser.add_argument("-mts", "--max-ts", type=int, default=2500, help="maximum local training set size")
    parser.add_argument("-m", "--method", type=str, default="sdv",
                        help="multilabel classifier to be used for type prediction")

    parser.add_argument("-sok", "--convert-to-sok", dest="sok", action="store_true",
                        help="convert csr_matrix to sok_matrix make cell access faster")
    parser.add_argument("-mc", "--mem-cache", dest="mem_cache", action="store_true",
                        help="use cache and evict to disk to reduce memory usage")
    parser.add_argument("-pp", "--plot", dest="plot", action="store_true", help="plot PRAUC")
    parser.add_argument("-ut", "--use-types", dest="use_types", action="store_true",
                        help="whether to use type assertions for learning embeddings")
    parser.set_defaults(mem_cache=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(use_types=False)
    parser.set_defaults(sok=False)

    args = parser.parse_args()

    method_name = args.method + (("-" + args.classifier) if args.method == "patybred" else "")
    if args.output is None:
        args.output = args.input.replace(".npz", "-ranked-facts-" + method_name + ".pkl")
    if args.save_path is None:
        args.save_path = args.input.replace(".npz", "-errdet-model-" + method_name + ".pkl")

    print(args)

    hist_path = args.input.replace(".npz", "-" + args.method + "-scores-dist.png")
    scores_path = args.input.replace(".npz", "-" + args.method + "-scores.pkl")

    d = np.load(args.input)

    X = d["data"]
    types = d["types"].item()
    domains = d["domains"].item()
    ranges = d["ranges"].item()
    type_hierarchy = None

    ents_dict = d["entities_dict"].item()
    types_dict = d["types_dict"].item()

    if not isinstance(ents_dict.keys()[0],int):
        ents_dict = {k:v for v,k in ents_dict.items()}
    if not isinstance(types_dict.keys()[0],int):
        types_dict = {k:v for v,k in types_dict.items()}


    X = [x if isinstance(x, sp.coo_matrix) else x.tocoo() for x in X]
    triples = to_triples(X, order="sop", dtype="list")

    n_relations = len(X)
    n_entities = X[0].shape[0]

    if args.method == "sdv":
        if args.load_path is None:
            ed = SDValidate()
        else:
            ed = SDValidate.load_model(args.load_path)
    if args.method == "bblbred":
        prefLabels = pickle.load(file(args.input.replace(".npz","-prefLabels-dict.pkl"),"rb"))
        definitions = pickle.load(file(args.input.replace(".npz", "-definitions-dict.pkl"), "rb"))
        ed = BblBRED(prefLabels=prefLabels, definitions=definitions, ents_dict=ents_dict, types_dict=types_dict,
                      max_depth=args.max_path_length, clf_name=args.classifier, so_type_feat=True,
                      n_neg=args.n_negatives, lfs=args.feature_selection, max_feats=args.max_feats,
                      min_sup=args.minimum_support, max_paths_per_level=args.max_paths_per_level,
                      path_selection_mode=args.path_selection_mode, reduce_mem_usage=args.mem_cache,
                      convert_to_sok=args.sok, max_pos_train=args.max_ts, max_fs_data_size=args.max_fs)
    if args.method == "pabred":
        ed = PaTyBRED(max_depth=args.max_path_length, clf_name=args.classifier, so_type_feat=False,
                      n_neg=args.n_negatives, lfs=args.feature_selection, max_feats=args.max_feats,
                      min_sup=args.minimum_support, max_paths_per_level=args.max_paths_per_level,
                      path_selection_mode=args.path_selection_mode, reduce_mem_usage=args.mem_cache,
                      convert_to_sok=args.sok, max_pos_train=args.max_ts, max_fs_data_size=args.max_fs)
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

    if args.load_path is None:
        t1 = datetime.now()
        ed.learn_model(X, types, type_hierarchy, domains, ranges)
        t2 = datetime.now()
        print("Training time = %f s" % (t2 - t1).total_seconds())
        if args.save_path is not None:
            ed.save_model(args.save_path)
    else:
        ed = ed.load_model(args.load_path)
        print("Model loaded from %s" % args.load_path)


    scores = ed.predict_proba(triples)

    scores = np.ravel(scores)
    id_rank = rankdata(scores)
    id_rank.sort()
    sorted_triples = [triples[i] for i in np.argsort(scores)]
    scores.sort()

    results = [(id_rank[i], scores[i], sorted_triples[i]) for i in range(len(scores))]
    pickle.dump(results, open(args.output, "wb"))

    if args.plot:
        plot_histogram(scores, title="Scores distribution (all facts)", fig_path="alltriples")
        rel_dict = d["relations_dict"].item()
        rel_dict = {k: v for v, k in rel_dict.items()}
        p_scores = [[] for p in range(n_relations)]
        for i, (s, o, p) in enumerate(triples):
            p_scores[p].append(scores[i])

        for p in range(n_relations):
            rel_name = short_str(rel_dict[p])
            print(rel_name, len(p_scores[p]))
            plot_histogram(p_scores[p], title="Scores distribution  (%s relation)" % rel_name,
                           fig_path="%s.png" % rel_name)
