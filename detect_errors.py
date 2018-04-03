import sys
from util import loadTypesNpz, loadGraphNpz, load_type_hierarchy, load_domains, load_ranges
from sdvalidate import SDValidate
from argparse import ArgumentParser
from patybred import PaTyBRED
from errordetector import OutlierErrorDetector
from embeddings import SKGEWrapper, ProjE
from util import to_triples
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from datetime import datetime


def filter_ranks(ranks):
    filtered_ranks = np.zeros(len(ranks))
    filtered = 0
    count = 0
    for i in range(len(ranks)):
        count += 1
        next_rank = ranks[i + 1] if (i + 1) < len(ranks) else float("inf")
        if next_rank > ranks[i]:
            filtered_ranks[(i - count + 1):(i + 1)] = ranks[i] - filtered - (count - 2.0) / 2
            filtered = filtered + count
            count = 0
    return filtered_ranks


def evaluate(scores, error_ids, plot_pr=False):
    print("%d facts with %d errors" % (scores.shape[0], len(error_ids)))

    scores = np.ravel(scores)
    id_rank = rankdata(scores)
    error_ranks = np.array([i for id, i in enumerate(id_rank) if id in error_ids])
    error_ranks.sort()
    mean_rank = error_ranks.mean()
    mrr = np.reciprocal(error_ranks).mean()

    filtered_error_ranks = filter_ranks(error_ranks)
    fmean_rank = filtered_error_ranks.mean()
    fmrr = np.reciprocal(filtered_error_ranks).mean()

    y = np.array([1 if i in error_ids else 0 for i in range(len(scores))])
    rocauc = roc_auc_score(y, -scores)
    p, r, ts = precision_recall_curve(y, -scores)
    prauc = auc(r, p)

    if plot_pr:
        plt.plot(r, p)
        plt.show()

    print("FMeanRank = %f \t FMRR = %f \t MeanRank = %f \t MRR = %f \t ROCAUC = %f \t PRAUC = %f" % (
        fmean_rank, fmrr, mean_rank, mrr, rocauc, prauc))

    return mean_rank, mrr, rocauc, prauc


if __name__ == '__main__':
    parser = ArgumentParser(description="Type prediction evalutation with cross-validation")
    parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
    parser.add_argument("-e", "--embeddings", type=str, default=None, help="path to dataset with hole embeddings")
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

    X = loadGraphNpz(args.input)
    types = loadTypesNpz(args.input)
    domains = load_domains(args.input)
    ranges = load_ranges(args.input)
    type_hierarchy = load_type_hierarchy(args.input)


    triples = to_triples(X, order="sop", dtype="list")

    errors = np.load(args.input)["errors"]
    errors = [tuple(t) for t in errors]
    error_ids = [i for i, e in enumerate(triples) if e in errors]

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

    if args.load_path is None:
        t1 = datetime.now()
        ed.learn_model(X, types, type_hierarchy, domains, ranges)
        t2 = datetime.now()
        print("Training time = %f s" % (t2 - t1).total_seconds())

    t1 = datetime.now()
    scores = ed.predict_proba(triples)
    t2 = datetime.now()
    print("Prediction time = %f" % (t2 - t1).total_seconds())

    evaluate(scores, error_ids, plot_pr=args.plot)

    if args.method in ["proje", "transe", "hole", "rescal"]:
        oed = OutlierErrorDetector(ed, method=args.outlier_detection_method, outlier_per_relation=True)
        oed.learn_model(X, types, type_hierarchy, domains, ranges)
        scores = oed.predict_proba(triples)
        evaluate(scores, error_ids, plot_pr=args.plot)
