import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from util import to_triples
from sklearn.ensemble import IsolationForest
import numpy as np


class ErrorDetector(object):
    def learn_model(self, X, types=None, type_hierarchy=None, domains=None, ranges=None):
        pass

    def predict_proba(self, triples):
        pass

    def predict(self, triples):
        pass

    def save_model(self, path):
        pass

    def compute_relation_scores(self, r):
        pass

    @staticmethod
    def load_model(path):
        pass

    @staticmethod
    def save_scores_histogram(output_path, scores, bins=50):
        n, bins, patches = plt.hist(scores, bins, range=(min(scores), max(scores)), normed=1, facecolor='green',
                                    alpha=0.75)
        plt.savefig(output_path)
        return n, bins, patches


class OutlierErrorDetector(ErrorDetector):
    def __init__(self, ed, method="sd", outlier_per_relation=True):
        self.ed = ed
        self.method = method
        self.outlier_per_relation = outlier_per_relation
        if outlier_per_relation:
            self.outlier_detector = []
            self.mean = []
            self.std = []
        else:
            self.outlier_detector = None
            self.mean = 0
            self.std = 0

    def create_outlier_detector(self):
        if self.method == "ee":
            return EllipticEnvelope()
        elif self.method == "1csvm":
            return OneClassSVM()
        elif self.method == "if":
            return IsolationForest()
        else:
            return None

    def learn_model(self, X, types=None, type_hierarchy=None, domains=None, ranges=None):
        if self.outlier_per_relation:
            for r in range(len(X)):
                if X[r].nnz > 1:
                    triples = zip(list(X[r].row), list(X[r].col), [r] * X[r].nnz)
                    scores = self.ed.predict_proba(triples)
                    self.mean.append(np.mean(scores))
                    self.std.append(np.std(scores))
                    od = self.create_outlier_detector()
                    scores.reshape((-1, 1))
                    if od is not None:
                        od.fit(scores)
                    self.outlier_detector.append(od)
                else:
                    self.outlier_detector.append(None)
                    self.mean.append(1.0)
                    self.std.append(0.0)
        else:
            triples = to_triples(X, order="sop")
            scores = self.ed.predict_proba(triples)
            scores.reshape((-1, 1))
            self.mean = np.mean(scores)
            self.std = np.std(scores)
            od = self.create_outlier_detector()
            if od is not None:
                od.fit(scores)
            self.outlier_detector = od

    def predict_proba(self, triples):
        scores = self.ed.predict_proba(triples)
        if not self.outlier_per_relation:
            if self.outlier_detector is not None:
                od_scores = self.outlier_detector.decision_function(scores).reshape((-1, 1))
            else:
                od_scores = scores
            od_scores = np.multiply(od_scores, (scores < self.mean).astype(float)) + (scores >= self.mean).astype(float)
            return od_scores
        else:
            od_scores = np.zeros((len(triples), 1))
            for i, t in enumerate(triples):
                scores_i = scores[i].reshape((1, -1))
                r = t[2]
                od = self.outlier_detector[r]
                if od is not None:
                    r_od_scores = od.decision_function(scores_i)
                else:
                    r_od_scores = (scores_i - self.mean[r]) / self.std[r] if self.std[r] else 1.0
                r_od_scores = (r_od_scores * (scores_i < self.mean[r]).astype(float)) + (
                    scores_i >= self.mean[r]).astype(float)
                od_scores[i, 0] += r_od_scores[0, 0]

            return od_scores

    def predict(self, triples):
        pass

    def save_model(self, path):
        pass
