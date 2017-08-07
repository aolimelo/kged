import numpy as np
from numpy import linalg
from errordetector import ErrorDetector
import pickle
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack


class SDValidate(ErrorDetector):
    def __init__(self):
        pass

    def cosine_similarity(self, a, b):
        return a.dot(b) / (linalg.norm(a, 2) * linalg.norm(b, 2))

    def add_dicts(self, a, b):
        for k, v in b.items():
            if k not in a:
                a[k] = v
            else:
                a[k] += v
        return a

    def inc_dict(self, x={}, indices=[], add=1):
        for i in indices:
            if i not in x:
                x[i] = add
            else:
                x[i] += add
        return x

    def dict_to_array(self, d, n_instances):
        x = np.zeros(n_instances, dtype=float)
        x[d.keys()] = d.values()
        return x

    def add_thing_if_absent(self, types):
        if not types.sum(axis=1).all():
            types = hstack((types, csr_matrix(np.ones((types.shape[0], 1)))))
        return types

    def learn_model(self, X, types, type_hierarchy=None, domains=None, ranges=None):
        types = self.add_thing_if_absent(types)
        self.types = coo_matrix(types).tocsr()
        self.n_instances = X[0].shape[0]
        self.n_relations = len(X)
        self.n_types = types.shape[1]
        self.shape = (self.n_instances, self.n_instances)
        for slice in X:
            assert slice.shape == self.shape
        self.X = X

        print("computing subject/object type distributions")
        self.compute_distributions1()
        print("computing rpf")
        self.compute_rpf1();

    def compute_rpf1(self):
        facts_o = {}
        facts_o_r = {}
        for r in range(self.n_relations):
            slice = self.X[r]
            facts_o_r[r] = {}
            facts_o_r[r] = self.inc_dict(facts_o_r[r], slice.col)
            facts_o = self.add_dicts(facts_o, facts_o_r[r])

        for r in range(self.n_relations):
            facts_o_r[r] = self.dict_to_array(facts_o_r[r], self.n_instances)
        facts_o = self.dict_to_array(facts_o, self.n_instances)

        self.p_r_o = [np.nan_to_num(facts_o_r[r] / facts_o) for r in range(self.n_relations)]

    def compute_distributions1(self):
        ot_dist = [{} for r in range(self.n_relations)]
        st_dist = [{} for r in range(self.n_relations)]
        count = 0
        for r in range(self.n_relations):
            slice = self.X[r]
            for s in slice.row:
                st_dist[r] = self.inc_dict(st_dist[r], self.types[s].indices)
            for o in slice.col:
                ot_dist[r] = self.inc_dict(ot_dist[r], self.types[o].indices)
            count += slice.nnz

        self.st_dist = {}
        self.ot_dist = {}
        for r in range(self.n_relations):
            self.st_dist[r] = self.dict_to_array(st_dist[r], self.n_types) / count
            self.ot_dist[r] = self.dict_to_array(ot_dist[r], self.n_types) / count

    def compute_scores(self):
        print("computing scores")
        scores = []
        for r in range(self.n_relations):
            slice = self.X[r]
            for o in slice.col:
                fact_ot = self.types[o, :].todense()
                cos_sim = self.cosine_similarity(fact_ot, self.ot_dist[r])
                scores.append(cos_sim[0, 0])

        return scores

    def predict_proba(self, triples):
        # print("computing probabilities")
        scores = []
        for s, o, p in triples:
            fact_ot = self.types[o, :].todense().astype(float)
            cos_sim = self.cosine_similarity(fact_ot, self.ot_dist[p])
            scores.append(cos_sim[0, 0])
        return np.array(scores).reshape((-1, 1))

    def predict(self, triples):
        return (self.predict_proba(triples) > 0.5).astype(float)

    def save_model(self, path):
        pickle.dump(self, file(path, "wb"))

    @staticmethod
    def load_model(path):
        ed = pickle.load(file(path, "rb"))
        return ed
