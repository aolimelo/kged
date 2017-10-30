import numpy as np
import sys
from scipy.sparse import csr_matrix, find, coo_matrix, spmatrix
from random import randint, random
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from numpy import argsort
from collections import defaultdict as ddict
from copy import deepcopy
from Queue import Queue
import tensorflow as tf
from tqdm import tqdm
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.
    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.
    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.
    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.
    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2
    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    for x in xrange(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


def get_deletes_list(w, max_edit_distance):
    '''given a word, derive strings with up to max_edit_distance characters
       deleted'''
    deletes = []
    queue = [w]
    for d in range(max_edit_distance):
        temp_queue = []
        for word in queue:
            if len(word) > 1:
                for c in range(len(word)):  # character index
                    word_minus_c = word[:c] + word[c + 1:]
                    if word_minus_c not in deletes:
                        deletes.append(word_minus_c)
                    if word_minus_c not in temp_queue:
                        temp_queue.append(word_minus_c)
        queue = temp_queue

    return deletes


def short_str(e):
    s = e.encode("utf8")
    offset = max([s.rfind("/"), s.rfind("#")])
    return s[offset + 1:].replace(",", "")


def is_symmetric(m):
    """Check if a sparse matrix is symmetric
    Parameters  m : array or sparse matrix (A square matrix).
    Returns     check : bool (The check result).
    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def loadGraphNpz(inputDir):
    dataset = np.load(inputDir)
    data = dataset["data"]
    if isinstance(data[0], spmatrix):
        return data.tolist()
        print('The number of tensor slices: %d' % len(data))
        print('The number of non-zero values in the tensor: %d' % sum([Xi.nnz for Xi in data]))
    else:
        shape = dataset["shape"]
        dim = shape[1]
        num_slices = shape[0]
        X = []
        numNonzeroTensorEntries = 0
        for slice in data:
            row = slice["rows"]
            col = slice["cols"]
            Xi = coo_matrix((np.ones(len(row)), (row, col)), shape=(dim, dim))
            X.append(Xi)
            numNonzeroTensorEntries += len(row)

        print('The number of tensor slices: %d' % num_slices)
        print('The number of non-zero values in the tensor: %d' % numNonzeroTensorEntries)
        return X


def loadTypesNpz(inputDir):
    dataset = np.load(inputDir)
    D_type = None
    if "typeshape" in dataset and "typedata" in dataset:
        typeshape = dataset["typeshape"]
        typedata = dataset["typedata"].item()
        row = typedata["rows"]
        col = typedata["cols"]
        val = typedata["vals"]
        D_type = coo_matrix((val, (row, col)), shape=typeshape)
        print('The number of non-zero values in the types matrix: %d' % D_type.nnz)
    if "types" in dataset:
        D_type = dataset["types"].item()
    return D_type


def load_domains(inputDir):
    dataset = np.load(inputDir)
    return dataset["domains"].item() if "domains" in dataset else None


def load_ranges(inputDir):
    dataset = np.load(inputDir)
    return dataset["ranges"].item() if "ranges" in dataset else None


def load_type_hierarchy(inputDir):
    dataset = np.load(inputDir)
    hierarchy = dataset["type_hierarchy"].item() if "type_hierarchy" in dataset else None
    if hierarchy is not None:
        for i, n in hierarchy.items():
            try:
                n.children = [hierarchy[c] for c in n.children]
                n.parents = [hierarchy[p] for p in n.parents]
            except:
                pass
    return hierarchy


def load_relations_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["relations_dict"].item() if "relations_dict" in dataset else None

def load_entities_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["entities_dict"].item() if "entities_dict" in dataset else None

def jaccard_index(s1, s2):
    return float(len(s1.intersection(s2))) / len(s1.union(s2))


def jaccard_distance(s1, s2):
    return 1.0 - jaccard_index(s1, s2)


def in_csr(m, i, j):
    return j in m.indices[m.indptr[i]:m.indptr[i + 1]]


def in_csc(m, i, j):
    return i in m.indices[m.indptr[j]:m.indptr[j + 1]]


# Set Of Keys matrix, read-only
class sok_matrix:
    def __init__(self, m):
        assert isinstance(m, coo_matrix)
        self.keys = set([(m.row[i], m.col[i]) for i in range(m.nnz)])
        self.nnz = m.nnz
        self.shape = m.shape

    def __getitem__(self, key):
        return key in self.keys

    def tocoo(self, copy=True):
        rows, cols = zip(*list(self.keys))
        return coo_matrix(([True] * self.nnz, (rows, cols)), shape=self.shape, dtype=bool)


class sos_matrix:
    def __init__(self, m):
        assert isinstance(m, coo_matrix)
        self.keys = ddict(lambda: [])
        for i in range(m.nnz):
            self.keys[m.row[i]].append(m.col[i])
        self.keys = {k: set(v) for k, v in self.keys.items()}
        self.nnz = m.nnz
        self.shape = m.shape

    def __getitem__(self, key):
        return key[0] in self.keys and key[1] in self.keys[key[0]]

    def tocoo(self, copy=True):
        rows, cols = [], []
        for k, v in self.keys.items():
            rows += [k] * len(v)
            cols += list(v)
        return coo_matrix(([True] * self.nnz, (rows, cols)), shape=self.shape, dtype=bool)


class bcsr_matrix(csr_matrix):
    def __getitem__(self, key):
        return key[1] in self.indices[self.indptr[key[0]]:self.indptr[key[0] + 1]]


class sli_matrix:
    def __init__(self, m):
        try:
            assert isinstance(m, coo_matrix)
            if sys.version_info > (3,):
                long = int
            self.keys = set([long(m.row[i]) * long(m.shape[1]) + m.col[i] for i in range(m.nnz)])
            self.nnz = m.nnz
            self.shape = m.shape
        except RuntimeWarning:
            raise RuntimeError("Overflow encountered in long_scalars")

    def __getitem__(self, key):
        return key[0] * self.shape[1] + key[1] in self.keys()

    def tocoo(self, copy=True):
        rows = [i // self.shape[1] for i in self.keys]
        cols = [i % self.shape[1] for i in self.keys]
        return coo_matrix(([True] * self.nnz, (rows, cols)), shape=self.shape, dtype=bool)


def ccorr(a, b):
    return ifft(np.conj(fft(a)) * fft(b)).real


def plot_histogram(data, bins=50, title="", xlabel="score", ylabel="count", fig_path=None):
    n, bins, patches = plt.hist(data, bins=bins, facecolor='gray')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()
    plt.close()


def to_triples(X, order="pso", dtype="array"):
    h, t, r = [], [], []
    for i in range(len(X)):
        Xi = X[i] if isinstance(X[i], coo_matrix) else X[i].tocoo()
        r.extend(np.full((X[i].nnz), i))
        h.extend(X[i].row.tolist())
        t.extend(X[i].col.tolist())
    if order == "spo":
        triples = zip(h, r, t)
    if order == "pso":
        triples = zip(r, h, t)
    if order == "sop":
        triples = zip(h, t, r)
    if dtype == "list":
        return triples
    return np.array(triples)


class DummyConstantPrecitor(object):
    def __init__(self, const=0.0):
        self.const = const

    def fit(self, X=None, y=None):
        pass

    def predict(self, X):
        return self.predict_proba(X).astype(int)

    def predict_proba(self, X):
        return np.full((X.shape[0], 1), self.const, dtype=float)


def lp_scores(mdl, xs, ys):
    scores = mdl.predict_proba(xs)
    pr, rc, _ = precision_recall_curve(ys, scores)
    roc = roc_auc_score(ys, scores)
    f1 = 2 * (np.multiply(pr, rc)) / (pr + rc)
    return auc(rc, pr), roc, np.max(np.nan_to_num(f1))


def ranking_scores(pos, fpos):
    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    fmrr = print_pos(
        np.array(hpos + tpos),
        np.array(fhpos + ftpos))
    return fmrr


def print_pos(pos, fpos):
    mrr, mean_pos, hits = compute_scores(pos)
    fmrr, fmean_pos, fhits = compute_scores(fpos)
    print(
        "MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@10 = %.2f/%.2f" %
        (mrr, fmrr, mean_pos, fmean_pos, hits, fhits)
    )
    return fmrr


def compute_scores(pos, hits=10):
    mrr = np.mean(1.0 / pos)
    print(pos)
    mean_pos = np.mean(pos)
    hits = np.mean(pos <= hits).sum() * 100
    return mrr, mean_pos, hits


def corrupt(triple, n_entities, X=None, p_corrupt_subj=0.5):
    s, o, p = triple
    s, p, o = int(s), int(p), int(o)
    if p_corrupt_subj == 0 or np.random.uniform() >= p_corrupt_subj:
        new_o = np.random.randint(n_entities)
        while X[p][s, new_o] > 0:
            new_o = np.random.randint(n_entities)
        return (s, new_o, p)
    else:
        new_s = np.random.randint(n_entities)
        while X[p][new_s, o] > 0:
            new_s = np.random.randint(n_entities)
        return (new_s, o, p)


def generate_random_negatives(triples, n_entities, X=None, p_corrupt_subj=0.5):
    if X is not None:
        assert n_entities == X[0].shape[0]
    negs_batch = [corrupt(triple, n_entities, X, p_corrupt_subj) for triple in triples]
    return negs_batch


def generate_negatives(n_entities, pos_pairs, all_pos_pairs=None, n_neg=1):
    if all_pos_pairs is None:
        all_pos_pairs = pos_pairs

    neg_pairs = []
    for s, o in pos_pairs:
        for i in range(n_neg):
            if random() > 0.5:
                s = randint(0, n_entities - 1)
            else:
                o = randint(0, n_entities - 1)
            neg_pair = tuple([s, o])
            if neg_pair not in all_pos_pairs and neg_pair not in neg_pairs:
                neg_pairs.append(tuple([s, o]))

    return neg_pairs


def generate_negatives_from_existing_so_pairs(n_entities, r_pos_pairs, all_pos_pairs, n_neg=1):
    neg_pairs = []
    all_s_o = ddict(lambda: [])
    all_o_s = ddict(lambda: [])
    for s, o in all_pos_pairs:
        all_s_o[s].append(o)
        all_o_s[o].append(s)
    for s, o in r_pos_pairs:
        for i in range(n_neg):
            while True:
                random_number = random()
                if random() < 0.01:
                    neg = randint(0, n_entities - 1)
                else:
                    candidates = all_s_o[s] if random_number > 0.5 else all_o_s[o]
                    neg = candidates[randint(0, len(candidates) - 1)]
                neg_pair = (s, neg) if random_number > 0.5 else (neg, o)

                if not (neg_pair in r_pos_pairs or neg_pair in neg_pairs):
                    break

            neg_pairs.append(neg_pair)

    return neg_pairs


def generate_negatives_pra(P_r, feat_paths, path_matrices, pos_pairs, exp_base=1.25):
    n_entities = P_r.shape[0]
    neg_pairs = []
    subjects = set([pair[0] for pair in pos_pairs])

    sample_index = lambda k: int(k * exp_base ** k)

    for pos_s in subjects:
        scores = csr_matrix((1, n_entities), dtype=float)
        for path in feat_paths:
            scores = scores + path_matrices[path][pos_s]

        non_zeros = find(scores)
        sorted_scores = sorted(zip(find(scores)[2], find(scores)[1]), reverse=True)
        k = 1
        while sample_index(k) < len(sorted_scores):
            i = sample_index(k)
            sc, neg_o = sorted_scores[i]

            while tuple([pos_s, neg_o]) in pos_pairs and (i + 1) < len(sorted_scores):
                i += 1
                sc, neg_o = sorted_scores[i]

            if i < len(sorted_scores):
                neg_pairs.append(tuple([pos_s, neg_o]))
                # print("pos_s=%d score>0 negpairs=%d i=%d k=%d" % (pos_s,len(neg_pairs),i, k))
            k += 1

        while sample_index(k) < n_entities:
            i = sample_index(k)
            neg_o = randint(0, n_entities - 1)
            while neg_o in non_zeros[1] or tuple([pos_s, neg_o]) in neg_pairs:
                neg_o += 1

            if i < n_entities and neg_o < n_entities:
                neg_pairs.append(tuple([pos_s, neg_o]))
                # print("pos_s=%d score=0 negpairs=%d i=%d k=%d" % (pos_s,len(neg_pairs), i, k))
            k += 1

    return neg_pairs


class FilteredRankingEval(object):
    def __init__(self, true_triples, xs, neval=-1):
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        at = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.sz = len(xs)
        for s, o, p in xs:
            idx[p].append((s, o))

        for s, o, p in true_triples:
            tt[p]['os'][s].append(o)
            tt[p]['ss'][o].append(s)

        for s, o, p in true_triples:
            at[p]['os'][s].append(o)
            at[p]['ss'][o].append(s)

        self.idx = dict(idx)
        self.tt = dict(tt)
        self.at = dict(at)

        self.neval = {}
        for p, sos in self.idx.items():
            if neval == -1:
                self.neval[p] = -1
            else:
                self.neval[p] = np.int(np.ceil(neval * len(sos) / len(xs)))

    def scores_o(self, mdl, s, p):
        try:
            return mdl.scores_o(s, p)
        except:
            triples = []
            for o in range(mdl.n_instances):
                triples.append((s, o, p))
            return mdl.predict_proba(triples)

    def scores_s(self, mdl, o, p):
        try:
            return mdl.scores_s(o, p)
        except:
            triples = []
            for s in range(mdl.n_instances):
                triples.append((s, o, p))
            return mdl.predict_proba(triples)

    def rank_o(self, mdl, s, p, o):
        scores_o = self.scores_o(mdl, s, p).flatten()
        if scores_o.shape[0] == 1 and scores_o.shape[0] < scores_o.shape[1]:
            scores_o = np.ravel(scores_o)
        # print("so=%s" % str(scores_o.shape))
        sortidx_o = argsort(scores_o)[::-1]
        rank = np.where(sortidx_o == o)[0][0] + 1

        rm_idx = self.at[p]['os'][s]
        rm_idx = [i for i in rm_idx if i != o]
        scores_o[rm_idx] = -np.Inf
        sortidx_o = argsort(scores_o)[::-1]
        frank = np.where(sortidx_o == o)[0][0] + 1
        return rank, frank

    def rank_s(self, mdl, s, p, o):
        scores_s = self.scores_s(mdl, o, p).flatten()
        if scores_s.shape[0] == 1 and scores_s.shape[0] < scores_s.shape[1]:
            scores_s = np.ravel(scores_s)
        # print("so=%s" % str(scores_o.shape))
        sortidx_o = argsort(scores_s)[::-1]
        rank = np.where(sortidx_o == o)[0][0] + 1

        rm_idx = self.at[p]['ss'][o]
        rm_idx = [i for i in rm_idx if i != s]
        scores_s[rm_idx] = -np.Inf
        sortidx_s = argsort(scores_s)[::-1]
        frank = np.where(sortidx_s == s)[0][0] + 1
        return rank, frank

    def positions(self, mdl):
        pos = {}
        fpos = {}

        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        for p, sos in self.idx.items():
            if p in self.tt:
                ppos = {'head': [], 'tail': []}
                pfpos = {'head': [], 'tail': []}

                if hasattr(self, 'prepare'):
                    self.prepare(mdl, p)

                for s, o in sos[:self.neval[p]]:
                    rank, frank = self.rank_o(mdl, s, p, o)
                    ppos['tail'].append(rank)
                    pfpos['tail'].append(frank)
                    rank, frank = self.rank_s(mdl, s, p, o)
                    ppos['head'].append(rank)
                    pfpos['head'].append(frank)
                pos[p] = ppos
                fpos[p] = pfpos

        return pos, fpos


class HolEEval(FilteredRankingEval):
    def prepare(self, mdl, p):
        self.ER = ccorr(mdl.R[p], mdl.E)

    def scores_o(self, mdl, s, p):
        return np.dot(self.ER, mdl.E[s])

    def scores_s(self, mdl, o, p):
        return np.dot(mdl.E, self.ER[o])


class TransEEval(FilteredRankingEval):
    def prepare(self, mdl, p):
        self.ER = mdl.E + mdl.R[p]

    def scores_o(self, mdl, s, p):
        return -np.sum(np.abs(self.ER[s] - mdl.E), axis=1)

    def scores_s(self, mdl, o, p):
        return -np.sum(np.abs(self.ER - mdl.E[o]), axis=1)


class RESCALEval(FilteredRankingEval):
    def prepare(self, mdl, p):
        self.EW = np.mat(mdl.E) * np.mat(mdl.W[p])

    def scores_o(self, mdl, s, p):
        return -np.sum(np.abs(self.EW[s] - mdl.E), axis=1)

    def scores_s(self, mdl, o, p):
        return -np.sum(np.abs(self.EW - mdl.E[o]), axis=1)


class RDF2VecEval(FilteredRankingEval):
    def prepare(self, mdl, p):
        pass

    def scores_o(self, mdl, s, p):
        if mdl.concat:
            return mdl.clfs[int(p)].predict_proba(
                np.hstack((np.tile(mdl.embs[int(s)], (mdl.embs.shape[0], 1)), mdl.embs)))
        else:
            return mdl.clfs[int(p)].predict_proba(mdl.embs - mdl.embs[int(s)])

    def scores_s(self, mdl, o, p):
        if mdl.concat:
            return mdl.clfs[int(p)].predict_proba(
                np.hstack((mdl.embs, np.tile(mdl.embs[int(o)], (mdl.embs.shape[0], 1)))))
        else:
            return mdl.clfs[int(p)].predict_proba(mdl.embs[int(o)] - mdl.embs)


class Emb2Huang03LPEval(FilteredRankingEval):
    def positions(self, mdl):
        self.all_s = set()
        self.all_o = set()
        for so_list in self.idx.values():
            for so in so_list:
                self.all_s.add(so[0])
                self.all_o.add(so[1])
        self.o_rank_values = {p: {s: {} for s in self.at[p]["os"].keys()} for p in self.at.keys()}
        self.s_rank_values = {p: {o: {} for o in self.at[p]["ss"].keys()} for p in self.at.keys()}
        self.o_filtered_rank_values = {p: {s: {} for s in self.at[p]["os"].keys()} for p in self.at.keys()}
        self.s_filtered_rank_values = {p: {o: {} for o in self.at[p]["ss"].keys()} for p in self.at.keys()}

        n = mdl.n_instances

        with mdl.sess.as_default():
            print("Predincting objects")
            pbar = tqdm(total=len(self.all_s))
            for s in self.all_s:
                ranks = mdl.rank.eval(
                    feed_dict={mdl.input_s: np.full((n, 1), s), mdl.input_o: np.arange(n).reshape(-1, 1)})
                for p in self.at.keys():
                    if s in self.at[p]["os"]:
                        for o in self.at[p]["os"][s]:
                            rank = np.where(ranks[p] == o)[0][0] + 1
                            self.o_rank_values[p][s][o] = rank
                        for i, (o, rank) in enumerate(sorted(self.o_rank_values[p][s].items(), key=lambda x: x[1])):
                            self.o_filtered_rank_values[p][s][o] = rank - i
                pbar.update(1)
            pbar.close()
            print("Predincting subjects")
            pbar = tqdm(total=len(self.all_s))
            for o in self.all_o:
                ranks = mdl.rank.eval(
                    feed_dict={mdl.input_s: np.arange(n).reshape(-1, 1), mdl.input_o: np.full((n, 1), o)})
                for p in self.at.keys():
                    if o in self.at[p]["ss"]:
                        for s in self.at[p]["ss"][o]:
                            rank = np.where(ranks[p] == s)[0][0] + 1
                            self.s_rank_values[p][o][s] = rank
                        for i, (s, rank) in enumerate(sorted(self.s_rank_values[p][o].items(), key=lambda x: x[1])):
                            self.s_filtered_rank_values[p][o][s] = rank - i
                pbar.update(1)
            pbar.close()
            return super(Emb2Huang03LPEval, self).positions(mdl)

    def prepare(self, mdl, p):
        pass

    def rank_o(self, mdl, s, p, o):
        return self.o_rank_values[p][s][o], self.o_filtered_rank_values[p][s][o]

    def rank_s(self, mdl, s, p, o):
        return self.s_rank_values[p][o][s], self.s_filtered_rank_values[p][o][s]


class RDF2VecSMEval(Emb2Huang03LPEval):
    def positions(self, mdl):
        self.all_s = set()
        self.all_o = set()
        for so_list in self.idx.values():
            for so in so_list:
                self.all_s.add(so[0])
                self.all_o.add(so[1])
        self.o_rank_values = {p: {s: {} for s in self.at[p]["os"].keys()} for p in self.at.keys()}
        self.s_rank_values = {p: {o: {} for o in self.at[p]["ss"].keys()} for p in self.at.keys()}
        self.o_filtered_rank_values = {p: {s: {} for s in self.at[p]["os"].keys()} for p in self.at.keys()}
        self.s_filtered_rank_values = {p: {o: {} for o in self.at[p]["ss"].keys()} for p in self.at.keys()}
        n = mdl.n_instances

        sess = tf.Session()

        for s in self.all_s:
            if mdl.concat:
                preds = mdl.clf.predict_proba(np.hstack((np.tile(mdl.embs[s], (mdl.embs.shape[0], 1)), mdl.embs)))
            else:
                preds = mdl.clf.predict_proba(mdl.embs - mdl.embs[s])
            for p in self.at.keys():
                if s in self.at[p]["os"]:
                    pred_p = preds[:, p]
                    sortidx_o = argsort(pred_p)[::-1]
                    for o in self.at[p]["os"][s]:
                        rank = np.where(sortidx_o == o)[0][0] + 1
                        self.o_rank_values[p][s][o] = rank
                    for i, (o, rank) in enumerate(sorted(self.o_rank_values[p][s].items(), key=lambda x: x[1])):
                        self.o_filtered_rank_values[p][s][o] = rank - i
        for o in self.all_o:
            if mdl.concat:
                preds = mdl.clf.predict_proba(np.hstack((mdl.embs, np.tile(mdl.embs[o], (mdl.embs.shape[0], 1)))))
            else:
                preds = mdl.clf.predict_proba(mdl.embs[o] - mdl.embs)
            for p in self.at.keys():
                if o in self.at[p]["ss"]:
                    pred_p = preds[:, p]
                    sortidx_s = argsort(pred_p)[::-1]
                    for s in self.at[p]["ss"][o]:
                        rank = np.where(sortidx_s == s)[0][0] + 1
                        self.s_rank_values[p][o][s] = rank
                    for i, (s, rank) in enumerate(sorted(self.s_rank_values[p][o].items(), key=lambda x: x[1])):
                        self.s_filtered_rank_values[p][o][s] = rank - i
        return super(Emb2Huang03LPEval, self).positions(mdl)


class PRAEval(FilteredRankingEval):
    def prepare(self, mdl, p):
        pass

    def scores_o(self, mdl, s, p):
        paths = mdl.feat_paths[p]
        feats = np.zeros((mdl.n_instances, len(paths)), dtype=float)
        for i, path in enumerate(paths):
            a = mdl.path_matrices[path]
            feats[:, i] = a[s].todense().flatten()
        if mdl.so_type_feat and mdl.types is not None:
            ss = [s] * mdl.n_instances
            oo = list(range(mdl.n_instances))
            feats = np.hstack((feats, mdl.types[ss].todense(), mdl.types[oo].todense()))
        if mdl.feat_selector[p] is not None:
            feats = mdl.feat_selector[p].transform(feats)
        if mdl.learn_weights:
            return mdl.models[p].predict_proba(feats)
        else:
            return feats.sum(axis=1).ravel((-1, 1))

    def scores_s(self, mdl, o, p):
        paths = mdl.feat_paths[p]
        feats = np.zeros((mdl.n_instances, len(paths)), dtype=float)
        for i, path in enumerate(paths):
            a = mdl.path_matrices[path]
            feats[:, i] = a[:, o].todense().flatten()
        if mdl.so_type_feat and mdl.types is not None:
            oo = [o] * mdl.n_instances
            ss = list(range(mdl.n_instances))
            feats = np.hstack((feats, mdl.types[ss].todense(), mdl.types[oo].todense()))
        if mdl.feat_selector[p] is not None:
            feats = mdl.feat_selector[p].transform(feats)
        if mdl.learn_weights:
            return mdl.models[p].predict_proba(feats)
        else:
            return feats.sum(axis=1).ravel((-1, 1))


class ProjEEval(FilteredRankingEval):
    def positions(self, mdl):
        self.all_s = set()
        self.all_o = set()
        for so_list in self.idx.values():
            for s, o in so_list:
                self.all_s.add(s)
                self.all_o.add(o)
        self.o_rank_values = {p: {s: {} for s in self.tt[p]["os"].keys()} for p in self.tt.keys()}
        self.s_rank_values = {p: {o: {} for o in self.tt[p]["ss"].keys()} for p in self.tt.keys()}
        self.o_filtered_rank_values = {p: {s: {} for s in self.tt[p]["os"].keys()} for p in self.tt.keys()}
        self.s_filtered_rank_values = {p: {o: {} for o in self.tt[p]["ss"].keys()} for p in self.tt.keys()}
        n = mdl.n_instances

        with mdl.sess.as_default():
            test_input = []
            for s in self.all_s:
                for p in self.tt.keys():
                    test_input.append((s, 0, p))
            ranks = mdl.tail_ids.eval({mdl.test_input: np.array(test_input)})
            for k, (s, p, _) in enumerate(test_input):
                if s in self.tt[p]["os"]:
                    for o in self.tt[p]["os"][s]:
                        rank = np.where(ranks[k] == o)[0] + 1
                        self.o_rank_values[p][s][o] = rank
                    for i, (o, rank) in enumerate(sorted(self.o_rank_values[p][s].items(), key=lambda x: x[1])):
                        self.o_filtered_rank_values[p][s][o] = rank - i

            test_input = []
            for o in self.all_o:
                for p in self.tt.keys():
                    test_input.append((0, o, p))
            ranks = mdl.head_ids.eval({mdl.test_input: np.array(test_input)})
            for k, (_, o, p) in enumerate(test_input):
                if o in self.tt[p]["ss"]:
                    for s in self.tt[p]["ss"][o]:
                        rank = np.where(ranks[k] == s)[0] + 1
                        self.s_rank_values[p][o][s] = rank
                    for i, (s, rank) in enumerate(sorted(self.s_rank_values[p][o].items(), key=lambda x: x[1])):
                        self.s_filtered_rank_values[p][o][s] = rank - i

            return super(ProjEEval, self).positions(mdl)

    def prepare(self, mdl, p):
        pass

    def rank_o(self, mdl, s, p, o):
        return self.o_rank_values[p][s][o], self.o_filtered_rank_values[p][s][o]

    def rank_s(self, mdl, s, p, o):
        return self.s_rank_values[p][o][s], self.s_filtered_rank_values[p][o][s]


class TransREval(FilteredRankingEval):
    def prepare(self, mdl, p):
        self.proj_embs = mdl.ent_embs * mdl.proj_rel

    def scores_o(self, mdl, s, p):
        return np.linalg.norm((self.proj_embs[s] + mdl.rel_emb[p] - self.proj_embs), ord=2, axis=1)

    def scores_s(self, mdl, o, p):
        return np.linalg.norm((self.proj_embs + mdl.rel_emb[p] - self.proj_embs[o]), ord=2, axis=1)


class ComplexEval(FilteredRankingEval):
    def positions(self, mdl):
        self.e1 = mdl.model.e1.eval()
        self.e2 = mdl.model.e2.eval()
        self.r1 = mdl.model.r1.eval()
        self.r2 = mdl.model.r2.eval()
        return super(ComplexEval, self).positions(mdl)

    def prepare(self, mdl, p):
        self.e1r1 = self.e1 * self.r1[p]
        self.e1r2 = self.e1 * self.r2[p]
        self.e2r1 = self.e2 * self.r1[p]
        self.e2r2 = self.e2 * self.r2[p]
        self.r1e1 = self.r1[p] * self.e1
        self.r1e2 = self.r1[p] * self.e2
        self.r2e1 = self.r2[p] * self.e1
        self.r2e2 = self.r2[p] * self.e2

    def scores_o(self, mdl, s, p):
        return np.sum(self.e1r1[s] * self.e1, axis=1) + \
               np.sum(self.e2r1[s] * self.e2, axis=1) + \
               np.sum(self.e1r2[s] * self.e2, axis=1) - \
               np.sum(self.e2r2[s] * self.e1, axis=1)

    def scores_s(self, mdl, o, p):
        return np.sum(self.e1 * self.r1e1[o], axis=1) + \
               np.sum(self.e2 * self.r1e2[o], axis=1) + \
               np.sum(self.e1 * self.r2e2[o], axis=1) - \
               np.sum(self.e2 * self.r2e1[o], axis=1)


def level_hierarchy(hier):
    if hier is None:
        return []
    roots = get_roots(hier)
    remaining = deepcopy(hier.keys())
    level = roots
    levels = []
    while level:
        next_level = []
        for n in level:
            for c in n.children:
                if isinstance(c,DAGNode):
                    if c.node_id in remaining:
                        next_level.append(c)
                        remaining.remove(c.node_id)
                else:
                    if c in remaining:
                        next_level.append(hier[c])
                        remaining.remove(c)

        levels.append(level)
        level = next_level
    return levels

def get_roots(hier):
    if not hier:
        return []
    else:
        roots = []
        for i, n in hier.items():
            if isinstance(n, DAGNode):
                if not n.parents:
                    roots.append(n)
            if isinstance(n, TreeNode):
                if n.parent is None:
                    roots.append(n)
        return roots


class TreeNode(object):
    def __init__(self, node_id, name, parent=None, children=[]):
        self.node_id = node_id
        self.name = name
        self.parent = parent
        self.children = children

    def __str__(self):
        return self.name.__str__()

    def print_tree(self, tab="", pool=None):
        print
        tab + self.name
        for child in self.children:
            if self != child and self == child.parent:
                child.print_tree(tab + "\t")

    def get_all_parents(self):
        parents = []
        nd = self
        while nd.parent is not None:
            parents.append(nd.parent)
            nd = nd.parent
        return parents

    def get_all_parent_ids(self):
        return [p.id for p in self.get_all_parents()]


class DAGNode(object):
    def __init__(self, node_id, name, parents=[], children=[]):
        self.node_id = node_id
        self.name = name
        self.parents = parents
        self.children = children

    def __str__(self):
        return self.name.__str__()

    def print_tree(self, tab="", pool=None):
        print(tab + self.name)
        for child in self.children:
            if self != child and self in child.parents:
                child.print_tree(tab + "\t")

    def get_all_parents(self):
        parents = set()
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            nd = queue.get()
            for p in nd.parents:
                if p not in parents:
                    parents.add(p)
                    queue.put(p)
        return parents

    def get_all_parent_ids(self):
        return [p.node_id for p in self.get_all_parents()]

    def to_tree(self):
        tree_node = TreeNode(self.node_id, self.name)
        tree_node.children = [c.to_tree for c in self.children]
        tree_node.parent = min(self.parents)
        return tree_node
