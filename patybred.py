import sys
import pickle
import gc
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, hstack
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from util import generate_negatives, in_csr, in_csc, sok_matrix, jaccard_index, jaccard_distance, is_symmetric, \
    to_triples, lazy_mult_matrix, lazy_matrix
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, SelectFromModel
from datetime import datetime
from sklearn.svm import SVC
from pympler import asizeof
import numpy as np
from errordetector import ErrorDetector
import signal
import warnings


class Model:
    pass

def handler(signum, frame):
    raise Exception("Path computation timeout")

class PaTyBRED(ErrorDetector):
    """
    Path Ranking algorithm
    %TODO add references
    """

    def __init__(self, rel_dict=None, alpha=0.01, max_depth=2, so_type_feat=True, so_iorels_feat=False,
                 single_model=False, n_neg=1,
                 lfs=None, max_feats=float("inf"), emb_model=None, min_sup=0.001, max_paths_per_level=float("inf"),
                 max_pos_train=2500,
                 learn_weights=True, debug=False, clf_name="lgr", path_selection_mode="m2", sparse_train_data=True,
                 max_fs_data_size=250,
                 reduce_mem_usage=False, convert_to_sok=False, lazy=False):
        self.alpha = alpha
        self.max_depth = max_depth
        self.rel_dict = rel_dict
        self.learn_weights = learn_weights
        self.debug = debug
        self.clf_name = clf_name
        self.so_type_feat = so_type_feat
        self.so_iorels_feat = so_iorels_feat
        self.single_model = single_model
        self.n_neg = n_neg
        self.lfs = lfs
        self.feat_selector = {}
        self.max_feats = max_feats
        self.emb_model = emb_model
        self.min_sup = min_sup
        self.max_paths_per_level = max_paths_per_level
        self.path_selection_mode = path_selection_mode
        self.max_pos_train = max_pos_train
        self.sparse_train_data = sparse_train_data
        self.max_fs_data_size = max_fs_data_size
        self.dump_mem = reduce_mem_usage
        self.selected_paths = {}
        self.selected_s_types, self.selected_o_types = {}, {}
        self.selected_out_s_feats, self.selected_out_o_feats, self.selected_in_s_feats, self.selected_in_o_feats = {}, {}, {}, {}
        self.n_selected_feats = {}
        self.matrix_paths = set()
        self.convert_to_sok = convert_to_sok
        self.max_nnz = 10000000
        self.timeout_secs = 600
        self.lazy = lazy

    def check_domain_range(self, r1, r2, domains, ranges, type_hierarchy):
        """
        Checks if range of r1 matches domain r2
        :param r1: relation 1
        :param r2: relation 2
        :param domains: dict of relations and domains
        :param ranges: dict of relations and ranges
        :param type_hierarchy:
        :return: true if range of r1 matches domain of r2, false otherwise
        """
        if domains is None or ranges is None:
            return True
        t1 = ranges[r1]
        t2 = domains[r2]
        if t1 is None or t2 is None or t1 == t2:
            return True
        elif type_hierarchy is not None:
            if t1 in type_hierarchy and t2 in type_hierarchy:
                n1 = type_hierarchy[t1]
                n2 = type_hierarchy[t2]
                if t1 in n2.get_all_parent_ids() or t2 in n1.get_all_parent_ids():
                    return True
        return False

    def so_any_intersection(self, path, r):
        s1 = self.path_rowscols[tuple(path)][1]
        s2 = self.path_rowscols[tuple([r])][0]
        if len(s2) > len(s1):
            s1, s2 = (s2, s1)
        for i in s1:
            if i in s2:
                return True
        return False

    def get_path_matrix(self, path):
        if not isinstance(path, tuple):
            path = tuple(path)
        return self.path_matrices[path]

    def add_path_matrix(self, path, m):
        if not isinstance(path, tuple):
            path = tuple(path)
        self.path_matrices[path] = m
        self.matrix_paths.add(path)

    def path_relevance(self, p1, r):
        o1 = self.path_rowscols[tuple(p1)][1]
        s2 = self.path_rowscols[tuple([r])][0]
        if self.path_selection_mode == "random":
            if self.so_any_intersection(p1, r):
                return 1
        elif self.path_selection_mode == "mult":
            if self.so_any_intersection(p1, r):
                return self.get_path_matrix(tuple(p1)).nnz * self.get_path_matrix(tuple([r])).nnz
        else:
            inter = o1.intersection(s2)
            if inter:
                if self.path_selection_mode == "inter":
                    return len(inter)
                if self.path_selection_mode.startswith("m"):
                    s1 = self.path_rowscols[tuple(p1)][0]
                    o2 = self.path_rowscols[tuple([r])][1]
                    if self.path_selection_mode == "m1":
                        return float(len(inter)) / (len(s1.intersection(o2)) + 1.0)
                    if self.path_selection_mode == "m2":
                        return len(inter) * len(s1.union(o2))


    def learn_model(self, X, types, type_hierarchy=None, domains=None, ranges=None):
        hash_id = (sum([xi.nnz for xi in X]) + bool(types is None) + bool(type_hierarchy is None) + bool(
            domains is None) + bool(ranges is None)) * len(X)

        self.path_matrices = {}

        self.n_instances = X[0].shape[0]
        self.n_relations = len(X)
        self.shape = (self.n_instances, self.n_instances)
        for slice in X:
            assert slice.shape == self.shape

        self.syms = []
        for r in range(self.n_relations):
            if is_symmetric(X[r]):
                self.syms.append(r)

        self.all_pos_pairs = set()
        for xi in X:
            self.all_pos_pairs = self.all_pos_pairs.union(set(zip(xi.row, xi.col)))
        self.all_pos_pairs = list(self.all_pos_pairs)
        self.X = [coo_matrix(Xi).astype(bool).tocsr() for Xi in X]

        if types is not None and not isinstance(types, csr_matrix):
            types = csr_matrix(coo_matrix(types))

        self.types = types
        self.n_types = types.shape[1] if types is not None else 1
        self.domains = domains if domains is not None else {}
        self.ranges = ranges if ranges is not None else {}
        for r in range(self.n_relations):
            if r not in self.domains:
                self.domains[r] = None
            if r not in self.ranges:
                self.ranges[r] = None
        self.type_hierarchy = type_hierarchy

        min_sup = float(self.n_instances) * self.min_sup
        print(min_sup)

        X = [self.X[r] for r in range(self.n_relations)] + [self.X[r].transpose() for r in range(self.n_relations)]

        self.relevant_relations = range(2 * self.n_relations)
        inverses = {r: (r + self.n_relations) for r in range(self.n_relations)}
        inverses.update({k: v for v, k in inverses.items()})

        for r in self.syms:
            del self.relevant_relations[self.relevant_relations.index(r + self.n_relations)]
        gc.collect()

        domains = self.domains
        ranges = self.ranges
        domains.update({inverses[r]: t for r, t in self.ranges.items()})
        ranges.update({inverses[r]: t for r, t in self.domains.items()})
        self.domains = domains
        self.ranges = ranges

        self.path_rowscols = {}
        singletons = []

        all_paths = []
        lp1_paths = set()
        l_paths = set()

        t1 = datetime.now()

        print("Computing paths adjacency matrices")
        matrices_size = 0
        for r in range(self.n_relations):
            if self.X[r].getnnz() and self.X[r].getnnz() >= min_sup:
                singletons.append(r)
                l_paths.add(tuple([r]))
                if inverses[r] in self.relevant_relations:
                    singletons.append(inverses[r])
                    l_paths.add(tuple([inverses[r]]))
            m = self.X[r].astype(bool)
            matrices_size += asizeof.asizeof(m)
            sys.stdout.write('\r%d' % matrices_size)
            sys.stdout.flush()
            self.add_path_matrix([r], m)
            self.add_path_matrix([inverses[r]], m.transpose())
            rows = set(np.where(self.X[r].indptr[1:] > self.X[r].indptr[:-1])[0])
            cols = set(self.X[r].indices)
            self.path_rowscols[tuple([r])] = (rows, cols)
            self.path_rowscols[tuple([inverses[r]])] = (cols, rows)

        depth = 1
        num_paths = len(singletons)

        computed_paths = self.n_relations * 2

        all_paths.append(list(l_paths))


        while depth < self.max_depth and l_paths:
            candidates = {}
            for path in l_paths:
                path_last_r = path[-1]

                for r2 in self.relevant_relations:
                    if path_last_r != inverses[r2] and (path_last_r != r2 or r2 not in self.syms) and \
                            self.check_domain_range(path_last_r, r2, domains, ranges, self.type_hierarchy):
                        new_path = list(path) + [r2]
                        if not tuple(new_path) in candidates and not tuple(
                                [inverses[i] for i in reversed(new_path)]) in candidates:
                            candidates[tuple(new_path)] = self.path_relevance(path, r2)

            if self.max_paths_per_level < len(candidates):
                if self.path_selection_mode == "random":
                    sorted_ids = np.random.choice(len(candidates), len(candidates), replace=False)
                else:
                    sorted_ids = np.argsort(candidates.values())
                selected = [candidates.keys()[sorted_ids[-(i + 1)]] for i in
                            range(min(self.max_paths_per_level, len(sorted_ids)))]
                print("top-%d paths selected out of %d candidates" % (len(selected), len(candidates)))
            else:
                selected = candidates.keys()

            pbar = tqdm(total=len(selected))
            signal.signal(signal.SIGALRM, handler)
            for new_path in selected:
                try:
                    path = new_path[:-1]
                    r2 = new_path[-1]
                    computed_paths += 2
                    A1 = self.get_path_matrix(path)
                    A2 = X[r2]
                    signal.alarm(self.timeout_secs)
                    if self.lazy:
                        prod = lazy_matrix(A1.dot(A2))
                    else:
                        prod = A1.dot(A2)
                    signal.alarm(0)

                    if prod.getnnz() and min_sup <= prod.getnnz() < self.max_nnz:
                        matrices_size += asizeof.asizeof(prod)
                        sys.stdout.write('\r%d' % matrices_size)
                        sys.stdout.flush()
                        new_path = list(path) + [r2]
                        lp1_paths.add(tuple(new_path))
                        lp1_paths.add(tuple([inverses[i] for i in reversed(new_path)]))
                        self.add_path_matrix(new_path, prod)
                        self.add_path_matrix([inverses[i] for i in reversed(new_path)], prod.transpose())
                        if self.so_iorels_feat or depth+1 < self.max_depth:
                            if self.lazy:
                                rows = set(np.where(prod.m.indptr[1:] > prod.m.indptr[:-1])[0])
                                cols = set(prod.m.indices)
                            else:
                                rows = set(np.where(prod.indptr[1:] > prod.indptr[:-1])[0])
                                cols = set(prod.indices)
                            self.path_rowscols[tuple(new_path)] = (rows, cols)
                            self.path_rowscols[tuple([inverses[i] for i in reversed(new_path)])] = (cols, rows)
                        num_paths += 1

                    pbar.update(1)
                except Exception, exc:
                    print(exc)

            pbar.close()
            all_paths.append(list(lp1_paths))
            l_paths = lp1_paths
            lp1_paths = set()
            depth += 1

        del X
        gc.collect()

        t2 = datetime.now()

        print("total paths = %d out of %d     [computed in %f s]" % (
            sum(len(l) for l in all_paths), computed_paths, (t2 - t1).total_seconds()))

        # print("converting to sok_matrix")
        if self.convert_to_sok:
            print("converting to sok_matrix")
            for p in self.matrix_paths:
                self.add_path_matrix(p, sok_matrix(self.get_path_matrix(p).tocoo()))
                # m = bcsr_matrix(self.get_path_matrix(p))
                # if len(p) > 1:
                #    m.data = None
                # self.add_path_matrix(p, m)

        self.path_domains = {}
        self.path_ranges = {}
        for paths in all_paths:
            for p in paths:
                self.path_domains[p] = domains[p[0]]
                self.path_ranges[p] = ranges[p[len(p) - 1]]

        self.domain_paths = {None: []}
        self.range_paths = {None: []}
        self.domain_paths.update({t: [] for t in range(self.n_types)})
        self.range_paths.update({t: [] for t in range(self.n_types)})
        for path, path_domain in self.path_domains.items():
            self.domain_paths[path_domain].append(path)
        for path, path_range in self.path_ranges.items():
            self.range_paths[path_range].append(path)

        print("Training relations local classifiers")
        self.learn_feature_weights()

        n_path_feats = sum([len(p) for p in self.selected_paths.values()])
        n_type_feats = sum([len(p) for p in self.selected_s_types.values()]) + sum(
            [len(p) for p in self.selected_o_types.values()])
        total_feats = n_path_feats + n_type_feats

        # print("Paths: %d/%d = %f" % (n_path_feats,total_feats,float(n_path_feats)/total_feats))
        # print("Types: %d/%d = %f" % (n_type_feats, total_feats, float(n_type_feats) / total_feats))
        # print("paths ", self.selected_paths)
        # print("s_types", self.selected_s_types)
        # print("o_types", self.selected_o_types)

    def learn_feature_weights(self):
        self.models = {}
        self.feat_paths = {}
        self.so_type_feat_list = {}
        self.so_iorels_feat_list = {}
        pbar = tqdm(total=self.n_relations)
        for r in range(self.n_relations):
            # print("learning weights of relation %d"%r)
            self.feat_paths[r], self.so_type_feat_list[r], self.so_iorels_feat_list[r], self.models[r] = \
                self.learn_relation_local_classifier(r, self.domains[r], self.ranges[r])
            pbar.update(1)
        pbar.close()

    def predict(self, triples):
        return self.predict_proba(triples, proba=False)

    def compute_scores(self):
        triples = to_triples(self.X, order="sop", type="list")
        return self.predict_proba(triples)

    def get_clf(self):
        if self.clf_name == "lgr":
            return LogisticRegression(penalty="l2", solver="liblinear", n_jobs=-1)
        if self.clf_name == "rf":
            return RandomForestClassifier()
        if self.clf_name == "dt":
            return DecisionTreeClassifier()
        if self.clf_name == "svm":
            return SVC(probability=True)
        if self.clf_name == "ee":
            return EllipticEnvelope()
        elif self.clf_name == "1csmv":
            return OneClassSVM()
        elif self.clf_name == "if":
            return IsolationForest()

    def select_features(self, r, feats, labels, feat_paths):
        n_types = self.n_types
        n_relations = self.n_relations
        n_paths = len(feat_paths)
        if feats.shape[1] > self.max_feats:
            if self.lfs == "tfs":
                fsclf = ExtraTreesClassifier()
                fsclf.fit(feats, labels)
                feat_selector = SelectFromModel(fsclf, prefit=True)
                # feat_selector.fit(feats, labels)
                indices = np.array(feat_selector.get_support(indices=True))
            else:
                if self.lfs == "chi2":
                    measure = chi2
                if self.lfs == "mi":
                    measure = mutual_info_classif
                feat_selector = SelectKBest(measure, k=self.max_feats)
                feat_selector.fit(feats, labels)
                indices = np.array(feat_selector.get_support(indices=True))
        else:
            indices = np.arange(feats.shape[1])
            feat_selector = None

        self.selected_paths[r] = [path for i, path in enumerate(feat_paths) if i in indices]

        offset = n_paths
        if self.so_type_feat and self.types is not None:
            self.selected_s_types[r] = [i for i in (indices - offset) if 0 <= i < n_types]
            self.selected_o_types[r] = [i for i in (indices - offset - n_types) if 0 <= i < n_types]
            offset += 2 * n_types

        if self.so_iorels_feat:
            feat_paths = list(feat_paths)
            self.selected_out_s_feats[r] = [feat_paths[i] for i in (indices - offset - 0 * n_paths) if 0 <= i < n_paths]
            self.selected_out_o_feats[r] = [feat_paths[i] for i in (indices - offset - 1 * n_paths) if 0 <= i < n_paths]
            self.selected_in_s_feats[r] = [feat_paths[i] for i in (indices - offset - 2 * n_paths) if 0 <= i < n_paths]
            self.selected_in_o_feats[r] = [feat_paths[i] for i in (indices - offset - 3 * n_paths) if 0 <= i < n_paths]

        self.n_selected_feats[r] = len(indices)

        return feat_selector

    def create_paths_table(self, paths, all_pairs):
        n_examples = len(all_pairs)
        n_feats = len(paths)
        if self.sparse_train_data:
            feats = lil_matrix((n_examples, n_feats), dtype=bool)
        else:
            feats = np.zeros((n_examples, n_feats), dtype=bool)
        for j, path in enumerate(paths):
            P = self.get_path_matrix(path)
            if self.convert_to_sok:
                for i, (s, o) in enumerate(all_pairs):
                    if P[s, o]:
                        feats[i, j] = P[s, o]
            else:
                if isinstance(P, csr_matrix):
                    for i, (s, o) in enumerate(all_pairs):
                        if in_csr(P, s, o):
                            feats[i, j] = True
                if isinstance(P, csc_matrix):
                    for i, (s, o) in enumerate(all_pairs):
                        if in_csc(P, s, o):
                            feats[i, j] = True
        if self.sparse_train_data:
            feats = feats.tocsr()
        return feats

    def create_types_table(self, all_pairs, selected_s_types, selected_o_types):
        ss = [p[0] for p in all_pairs]
        oo = [p[1] for p in all_pairs]
        if self.sparse_train_data:
            feats = hstack((self.types[ss][:, selected_s_types].astype(bool),
                            self.types[oo][:, selected_o_types].astype(bool)))
        else:
            feats = np.hstack((self.types[ss][:, selected_s_types].astype(bool).todense(),
                               self.types[oo][:, selected_o_types].astype(bool).todense()))
        return feats

    def create_iorels_table(self, all_pairs, selected_out_s_feats, selected_out_o_feats, selected_in_s_feats,
                            selected_in_o_feats):
        outgoing_s_feats, outgoing_o_feats, ingoing_s_feats, ingoing_o_feats = [], [], [], []
        for path in selected_out_s_feats:
            outgoing_s_feats.append([p[0] in self.path_rowscols[path][0] for p in all_pairs])
        for path in selected_out_o_feats:
            outgoing_o_feats.append([p[1] in self.path_rowscols[path][0] for p in all_pairs])
        for path in selected_in_s_feats:
            ingoing_s_feats.append([p[0] in self.path_rowscols[path][1] for p in all_pairs])
        for path in selected_in_o_feats:
            ingoing_o_feats.append([p[1] in self.path_rowscols[path][1] for p in all_pairs])
        outgoing_s_feats = np.array(outgoing_s_feats).transpose().reshape((len(all_pairs), -1))
        outgoing_o_feats = np.array(outgoing_o_feats).transpose().reshape((len(all_pairs), -1))
        ingoing_s_feats = np.array(ingoing_s_feats).transpose().reshape((len(all_pairs), -1))
        ingoing_o_feats = np.array(ingoing_o_feats).transpose().reshape((len(all_pairs), -1))
        io_rel_feats = np.hstack((outgoing_s_feats, outgoing_o_feats, ingoing_s_feats, ingoing_o_feats))
        io_rel_feats.astype(bool)
        if self.sparse_train_data:
            csr_matrix(io_rel_feats)

        return io_rel_feats

    def create_feats_table(self, all_pairs, paths, s_t, o_t, out_s, out_o, in_s, in_o):
        feats = self.create_paths_table(paths, all_pairs)
        # Add type features
        if self.so_type_feat and self.types is not None:
            type_feats = self.create_types_table(all_pairs, s_t, o_t)
            feats = hstack((feats, type_feats)) if self.sparse_train_data else np.hstack((feats, type_feats))
        # Add ingoing and outgoing relations features
        if self.so_iorels_feat:
            io_rel_feats = self.create_iorels_table(all_pairs, out_s, out_o, in_s, in_o)
            feats = hstack((feats, io_rel_feats)) if self.sparse_train_data else np.hstack((feats, io_rel_feats))
        return feats

    def create_training_data(self, r, all_pos_pairs, max_size, n_neg):
        if len(all_pos_pairs) > max_size:
            pos_pairs = [all_pos_pairs[i] for i in np.random.choice(len(all_pos_pairs), max_size, replace=False)]
        else:
            pos_pairs = all_pos_pairs
        neg_pairs = generate_negatives(self.n_instances, pos_pairs, all_pos_pairs, n_neg=n_neg)
        all_pairs = pos_pairs + neg_pairs
        n_positives = len(pos_pairs)
        n_negatives = len(neg_pairs)

        labels = np.vstack((np.ones((n_positives, 1)), np.zeros((n_negatives, 1), dtype=int))).ravel()
        feats = self.create_feats_table(all_pairs, self.selected_paths[r],
                                        self.selected_s_types[r], self.selected_o_types[r],
                                        self.selected_out_s_feats[r], self.selected_out_o_feats[r],
                                        self.selected_in_s_feats[r], self.selected_in_o_feats[r])
        return feats, labels

    def scores_s(self, o, p):
        ss = range(self.n_instances)
        oo = [o] * self.n_instances
        feats = self.create_feats_table(zip(ss, oo), self.selected_paths[p],
                                        self.selected_s_types[p], self.selected_o_types[p],
                                        self.selected_out_s_feats[p], self.selected_out_o_feats[p],
                                        self.selected_in_s_feats[p], self.selected_in_o_feats[p])
        return self.models[p].predict_proba(feats)[:, 1]

    def scores_o(self, s, p):
        oo = range(self.n_instances)
        ss = [s] * self.n_instances
        feats = self.create_feats_table(zip(ss, oo), self.selected_paths[p],
                                        self.selected_s_types[p], self.selected_o_types[p],
                                        self.selected_out_s_feats[p], self.selected_out_o_feats[p],
                                        self.selected_in_s_feats[p], self.selected_in_o_feats[p])
        return self.models[p].predict_proba(feats)[:, 1]

    def predict_proba(self, triples, proba=True):
        scores = []
        if self.emb_model is not None:
            emb_scores = self.emb_model.predict_proba(triples)
        for i, (s, o, p) in enumerate(triples):
            s, p, o = int(s), int(p), int(o)
            if self.models[p] is None:
                pred = np.zeros((1,), dtype=float)
            else:
                if self.learn_weights and p in self.models and self.models[p] is not None:
                    feats = self.create_feats_table([(s, o)], self.selected_paths[p],
                                                    self.selected_s_types[p], self.selected_o_types[p],
                                                    self.selected_out_s_feats[p], self.selected_out_o_feats[p],
                                                    self.selected_in_s_feats[p], self.selected_in_o_feats[p])

                    if proba:
                        if self.clf_name in ["ee", "lcsvm", "if"]:
                            pred = self.models[p].decision_function(feats)
                        else:
                            pred = self.models[p].predict_proba(feats)[:, 1]
                    else:
                        pred = self.models[p].predict(feats)
                else:
                    pred = np.sum(feats, axis=1).astype(float) / len(self.feat_paths[p])
            pred = pred.ravel()
            scores += pred.tolist()

        return np.array(scores).reshape((-1, 1))

    def learn_relation_local_classifier(self, r, domains, ranges):
        path_r = tuple([r])

        if self.max_depth == 0:
            feat_paths = []
        else:
            same_domain_paths = set(self.domain_paths[domains])
            same_domain_paths = same_domain_paths.union(set(self.domain_paths[None]))
            same_range_paths = set(self.range_paths[ranges])
            same_range_paths = same_range_paths.union(set(self.range_paths[None]))
            feat_paths = same_domain_paths.intersection(same_range_paths)
            if path_r in feat_paths:
                feat_paths.remove(path_r)

        so_type_feat_list = None
        so_iorel_feat_list = None

        all_relations = list(range(self.n_relations))
        if not self.so_type_feat:
            self.selected_s_types = {i: [] for i in all_relations}
            self.selected_o_types = {i: [] for i in all_relations}
        if not self.so_iorels_feat:
            self.selected_out_s_feats = {i: [] for i in all_relations}
            self.selected_out_o_feats = {i: [] for i in all_relations}
            self.selected_in_s_feats = {i: [] for i in all_relations}
            self.selected_in_o_feats = {i: [] for i in all_relations}

        if self.learn_weights:  # and feat_paths:# and path_r in self.path_matrices:
            P_r = self.get_path_matrix(path_r)
            if P_r.nnz:
                P_r_coo = P_r.tocoo(copy=True)
                all_pos_pairs = zip(P_r_coo.row, P_r_coo.col)
                if len(all_pos_pairs) > self.max_fs_data_size:
                    pos_pairs = [all_pos_pairs[i] for i in
                                 np.random.choice(len(all_pos_pairs), self.max_fs_data_size, replace=False)]
                else:
                    pos_pairs = all_pos_pairs
                neg_pairs = []
                if not self.clf_name in ["ee", "lcsvm", "if"]:
                    neg_pairs = generate_negatives(self.n_instances, pos_pairs, all_pos_pairs, n_neg=self.n_neg)
                    # neg_pairs = generate_negatives_from_existing_so_pairs(self.n_instances, pos_pairs, self.all_pos_pairs)

                n_positives = len(pos_pairs)
                n_negatives = len(neg_pairs)
                all_pairs = pos_pairs + neg_pairs

                labels = np.vstack((np.ones((n_positives, 1)), np.zeros((n_negatives, 1)))).ravel()
                all_t = list(range(self.n_types))
                feats = self.create_feats_table(all_pairs, feat_paths, all_t, all_t, feat_paths, feat_paths, feat_paths,
                                                feat_paths)
                self.feat_selector[r] = self.select_features(r, feats, labels, feat_paths)

                if len(pos_pairs) < len(all_pos_pairs):
                    feats, labels = self.create_training_data(r, all_pos_pairs, max_size=self.max_pos_train,
                                                              n_neg=self.n_neg)
                else:
                    if self.feat_selector[r] is not None:
                        feats = self.feat_selector[r].transform(feats)

                clf = self.get_clf()
                if self.clf_name in ["ee", "lcsvm", "if"]:
                    clf.fit(feats)
                else:
                    clf.fit(feats, labels)

                return feat_paths, so_type_feat_list, so_iorel_feat_list, clf
            else:
                return set(), None, None, None
        else:
            return feat_paths, None, None, None

    def save_model(self, path):
        if not path.endswith(".pkl"):
            path += ".pkl"

        all_selected_paths = []
        for paths in self.selected_paths.values():
            all_selected_paths += paths
        for p in self.path_matrices.keys():
            if p not in all_selected_paths:
                del self.path_matrices[p]

        # Save light model with no precomputed adjacency matrices
        m = Model()
        m.models = self.models
        m.selected_s_types = self.selected_s_types
        m.selected_o_types = self.selected_o_types
        m.selected_paths = self.selected_paths
        m.so_iorels_feat = self.so_iorels_feat
        m.selected_out_s_feats = self.selected_out_s_feats
        m.selected_out_o_feats = self.selected_out_o_feats
        m.selected_in_s_feats = self.selected_in_s_feats
        m.selected_in_o_feats = self.selected_in_o_feats
        pickle.dump(m, file(path.replace(".pkl", "-light.pkl"), "wb"))

        # Save full model
        pickle.dump(self, file(path, "wb"))
