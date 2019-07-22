import math
import os
from util import to_triples
from multiprocessing import Process, active_children
import tensorflow as tf
import timeit
from errordetector import ErrorDetector
from skge import StochasticTrainer, PairwiseStochasticTrainer, HolE, TransE, RESCAL
from skge import activation_functions as afs
import numpy as np
from skge.util import ccorr
from skge.sample import LCWASampler, CorruptedSampler, RandomModeSampler, type_index


def test_ops(model):
    test_input = tf.placeholder(tf.int32, [None, 3])
    head_ids, tail_ids, hrt_res, trh_res, triple_score = model.test(test_input)

    return test_input, head_ids, tail_ids, hrt_res, trh_res, triple_score


def worker_func(in_queue, out_queue, hr_t, tr_h):
    while True:
        dat, in_queue = in_queue[0], in_queue[1:]
        if dat is None:
            in_queue.task_done()
            continue

        testing_data, head_pred, tail_pred = dat
        out_queue.append(test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h))
        in_queue.task_done()


def data_generator_func(in_queue, out_queue, tr_h, hr_t, n_entity, neg_weight):
    while True:
        dat, in_queue = in_queue[0], in_queue[1:]
        if dat is None:
            break

        hr_tlist = list()
        hr_tweight = list()
        tr_hlist = list()
        tr_hweight = list()

        htr = dat

        for idx in range(htr.shape[0]):
            if np.random.uniform(-1, 1) > 0:  # t r predict h
                tr_hweight.append(
                    [1. if x in tr_h[htr[idx, 1]][htr[idx, 2]] else y for
                     x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])
                tr_hlist.append([htr[idx, 1], htr[idx, 2]])
            else:  # h r predict t
                hr_tweight.append(
                    [1. if x in hr_t[htr[idx, 0]][htr[idx, 2]] else y for
                     x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])

                hr_tlist.append([htr[idx, 0], htr[idx, 2]])

        out_queue.append((np.asarray(hr_tlist, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32),
                       np.asarray(tr_hlist, dtype=np.int32), np.asarray(tr_hweight, dtype=np.float32)))


def test_evaluation(testing_data, head_pred, tail_pred, hr_t, tr_h):
    assert len(testing_data) == len(head_pred)
    assert len(testing_data) == len(tail_pred)

    mean_rank_h = list()
    mean_rank_t = list()
    filtered_mean_rank_h = list()
    filtered_mean_rank_t = list()

    for i in range(len(testing_data)):
        h = testing_data[i, 0]
        t = testing_data[i, 1]
        r = testing_data[i, 2]
        # mean rank

        mr = 0
        for val in head_pred[i]:
            if val == h:
                mean_rank_h.append(mr)
                break
            mr += 1

        mr = 0
        for val in tail_pred[i]:
            if val == t:
                mean_rank_t.append(mr)
            mr += 1

        # filtered mean rank
        fmr = 0
        for val in head_pred[i]:
            if val == h:
                filtered_mean_rank_h.append(fmr)
                break
            if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
                continue
            else:
                fmr += 1

        fmr = 0
        for val in tail_pred[i]:
            if val == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1

    return (mean_rank_h, filtered_mean_rank_h), (mean_rank_t, filtered_mean_rank_t)


def train_ops(model, learning_rate=0.1, optimizer_str='gradient', regularizer_weight=1.0):
    # with tf.device('/gpu'):
    train_hrt_input = tf.placeholder(tf.int32, [None, 2])
    train_hrt_weight = tf.placeholder(tf.float32, [None, model.n_entity])
    train_trh_input = tf.placeholder(tf.int32, [None, 2])
    train_trh_weight = tf.placeholder(tf.float32, [None, model.n_entity])

    loss = model.train([train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight],
                       regularizer_weight=regularizer_weight)
    if optimizer_str == 'gradient':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_str == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer_str == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise NotImplementedError("Does not support %s optimizer" % optimizer_str)

    grads = optimizer.compute_gradients(loss, model.trainable_variables)

    op_train = optimizer.apply_gradients(grads)

    return train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight, loss, op_train


def gen_hr_t(triple_data):
    hr_t = dict()
    for h, t, r in triple_data:
        if h not in hr_t:
            hr_t[h] = dict()
        if r not in hr_t[h]:
            hr_t[h][r] = set()
        hr_t[h][r].add(t)

    return hr_t


def gen_tr_h(triple_data):
    tr_h = dict()
    for h, t, r in triple_data:
        if t not in tr_h:
            tr_h[t] = dict()
        if r not in tr_h[t]:
            tr_h[t][r] = set()
        tr_h[t][r].add(h)
    return tr_h


class ProjE(ErrorDetector):
    @property
    def n_train(self):
        return self.train_triples.shape[0]

    @property
    def trainable_variables(self):
        return self.trainable

    def training_data(self, batch_size=100):

        n_triple = len(self.train_triples)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            hr_tlist, hr_tweight, tr_hlist, tr_hweight = self.corrupted_training(
                self.train_triples[rand_idx[start:end]])
            yield hr_tlist, hr_tweight, tr_hlist, tr_hweight
            start = end

    def raw_training_data(self, batch_size=100):
        n_triple = len(self.train_triples)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.train_triples[rand_idx[start:end]]
            start = end

    def corrupted_training(self, htr):
        # [head(tail), relation, #of_total_positive_candidates, positive_instances..., negative_instances...]
        hr_tlist = list()
        hr_tweight = list()
        tr_hlist = list()
        tr_hweight = list()

        for idx in range(htr.shape[0]):
            if np.random.uniform(-1, 1) > 0:  # t r predict h
                tr_hweight.append(
                    [1. if x in self.train_tr_h[htr[idx, 1]][htr[idx, 2]] else 0. for x in range(self.n_entity)])
                tr_hlist.append([htr[idx, 1], htr[idx, 2]])
            else:  # h r predict t
                hr_tweight.append(
                    [1. if x in self.train_hr_t[htr[idx, 0]][htr[idx, 2]] else 0. for x in range(self.n_entity)])

                hr_tlist.append([htr[idx, 0], htr[idx, 2]])

        return np.asarray(hr_tlist, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32), \
               np.asarray(tr_hlist, dtype=np.int32), np.asarray(tr_hweight, dtype=np.float32)

    def __init__(self, embed_dim, n_relations, n_entities, combination_method='simple', dropout=0.5, neg_weight=0.25,
                 max_iter=100, save_per=0, verbose=False, add_types=False):

        if combination_method.lower() not in ['simple', 'matrix']:
            raise NotImplementedError("ProjE does not support using %s as combination method." % combination_method)

        self.combination_method = combination_method

        self.embed_dim = embed_dim
        self.initialized = False

        self.trainable = list()
        self.dropout = dropout
        self.n_instances = self.n_entity = n_entities
        self.n_relations = n_relations

        self.lr = 0.01
        self.optimizer = "adam"
        self.loss_weight = 1e-5
        self.n_generator = 10
        self.neg_weight = neg_weight
        self.n_worker = 3
        self.eval_batch = 500
        self.batch = 200
        self.max_iter = max_iter
        self.save_per = save_per
        self.eval_per = 5

        self.add_types = add_types
        self.verbose = verbose

    @staticmethod
    def load_model(path, X, dim):
        projewrap = ProjEED(embed_dim=dim, n_relations=len(X), n_entities=X[0].shape[0], max_iter=0)
        projewrap.learn_model(X)
        projewrap.load_trained_model(path)
        return projewrap

    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)
        print("Model saved at %s" % save_path)

    def load_trained_model(self, path):
        self.saver.restore(self.sess, path)

    def create_model(self):

        embed_dim = self.embed_dim

        bound = 6 / math.sqrt(embed_dim)

        # with tf.device('/gpu'):
        self.ent_embedding = tf.get_variable("ent_embedding", [self.n_entity, embed_dim],
                                             initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                       maxval=bound,
                                                                                       seed=345))
        self.trainable.append(self.ent_embedding)

        self.rel_embedding = tf.get_variable("rel_embedding", [self.n_relations, embed_dim],
                                             initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                       maxval=bound,
                                                                                       seed=346))
        self.trainable.append(self.rel_embedding)

        if self.combination_method.lower() == 'simple':
            self.hr_weighted_vector = tf.get_variable("simple_hr_combination_weights", [embed_dim * 2],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound,
                                                                                                seed=445))
            self.tr_weighted_vector = tf.get_variable("simple_tr_combination_weights", [embed_dim * 2],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound,
                                                                                                seed=445))
            self.trainable.append(self.hr_weighted_vector)
            self.trainable.append(self.tr_weighted_vector)
            self.hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                       initializer=tf.zeros([embed_dim]))
            self.tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                       initializer=tf.zeros([embed_dim]))

            self.trainable.append(self.hr_combination_bias)
            self.trainable.append(self.tr_combination_bias)

        else:
            self.hr_combination_matrix = tf.get_variable("matrix_hr_combination_layer",
                                                         [embed_dim * 2, embed_dim],
                                                         initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                   maxval=bound,
                                                                                                   seed=555))
            self.tr_combination_matrix = tf.get_variable("matrix_tr_combination_layer",
                                                         [embed_dim * 2, embed_dim],
                                                         initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                   maxval=bound,
                                                                                                   seed=555))
            self.trainable.append(self.hr_combination_matrix)
            self.trainable.append(self.tr_combination_matrix)
            self.hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                       initializer=tf.zeros([embed_dim]))
            self.tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                       initializer=tf.zeros([embed_dim]))

            self.trainable.append(self.hr_combination_bias)
            self.trainable.append(self.tr_combination_bias)

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.multiply(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor * tf.abs(weights), 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp * tf.abs(weights), 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.

    def train(self, inputs, regularizer_weight=1., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.initialized:
                scp.reuse_variables()
            rel_embedding = self.rel_embedding
            normalized_ent_embedding = self.ent_embedding

            hr_tlist, hr_tlist_weight, tr_hlist, tr_hlist_weight = inputs

            # (?, dim)
            hr_tlist_h = tf.nn.embedding_lookup(normalized_ent_embedding, hr_tlist[:, 0])
            hr_tlist_r = tf.nn.embedding_lookup(rel_embedding, hr_tlist[:, 1])
            # (?, dim)
            tr_hlist_t = tf.nn.embedding_lookup(normalized_ent_embedding, tr_hlist[:, 0])
            tr_hlist_r = tf.nn.embedding_lookup(rel_embedding, tr_hlist[:, 1])

            if self.combination_method.lower() == 'simple':

                # shape (?, dim)
                hr_tlist_hr = hr_tlist_h * self.hr_weighted_vector[:self.embed_dim] + \
                              hr_tlist_r * self.hr_weighted_vector[self.embed_dim:]

                hrt_res = tf.matmul(tf.nn.dropout(tf.tanh(hr_tlist_hr + self.hr_combination_bias), self.dropout),
                                    self.ent_embedding, transpose_b=True)

                tr_hlist_tr = tr_hlist_t * self.tr_weighted_vector[
                                           :self.embed_dim] + tr_hlist_r * self.tr_weighted_vector[
                                                                           self.embed_dim:]

                trh_res = tf.matmul(tf.nn.dropout(tf.tanh(tr_hlist_tr + self.tr_combination_bias), self.dropout),
                                    self.ent_embedding,
                                    transpose_b=True)

                self.regularizer_loss = regularizer_loss = tf.reduce_sum(
                    tf.abs(self.hr_weighted_vector)) + tf.reduce_sum(tf.abs(
                    self.tr_weighted_vector)) + tf.reduce_sum(tf.abs(self.ent_embedding)) + tf.reduce_sum(
                    tf.abs(self.rel_embedding))

            else:

                hr_tlist_hr = tf.nn.dropout(tf.tanh(tf.matmul(tf.concat(1, [hr_tlist_h, hr_tlist_r]),
                                                              self.hr_combination_matrix) + self.hr_combination_bias),
                                            self.dropout)

                hrt_res = tf.matmul(hr_tlist_hr, self.ent_embedding, transpose_b=True)

                tr_hlist_tr = tf.nn.dropout(tf.tanh(tf.matmul(tf.concat(1, [tr_hlist_t, tr_hlist_r]),
                                                              self.tr_combination_matrix) + self.tr_combination_bias),
                                            self.dropout)

                trh_res = tf.matmul(tr_hlist_tr, self.ent_embedding, transpose_b=True)

                self.regularizer_loss = regularizer_loss = tf.reduce_sum(
                    tf.abs(self.hr_combination_matrix)) + tf.reduce_sum(tf.abs(
                    self.tr_combination_matrix)) + tf.reduce_sum(tf.abs(self.ent_embedding)) + tf.reduce_sum(
                    tf.abs(self.rel_embedding))

            self.hrt_softmax = hrt_res_softmax = self.sampled_softmax(hrt_res, hr_tlist_weight)

            hrt_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(hrt_res_softmax, 1e-10, 1.0)) * tf.maximum(0.,
                                                                                   hr_tlist_weight))

            self.trh_softmax = trh_res_softmax = self.sampled_softmax(trh_res, tr_hlist_weight)
            trh_loss = -tf.reduce_sum(
                tf.log(tf.clip_by_value(trh_res_softmax, 1e-10, 1.0)) * tf.maximum(0., tr_hlist_weight))
            return hrt_loss + trh_loss + regularizer_loss * regularizer_weight

    def test(self, inputs, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            h = tf.nn.embedding_lookup(self.ent_embedding, inputs[:, 0])
            t = tf.nn.embedding_lookup(self.ent_embedding, inputs[:, 1])
            r = tf.nn.embedding_lookup(self.rel_embedding, inputs[:, 2])

            ent_mat = tf.transpose(self.ent_embedding)

            if self.combination_method.lower() == 'simple':

                # predict tails
                hr = h * self.hr_weighted_vector[:self.embed_dim] + r * self.hr_weighted_vector[self.embed_dim:]
                hrt_res = tf.matmul(tf.tanh(hr + self.hr_combination_bias), ent_mat)
                _, tail_ids = tf.nn.top_k(hrt_res, k=self.n_entity)

                # predict heads
                tr = t * self.tr_weighted_vector[:self.embed_dim] + r * self.tr_weighted_vector[self.embed_dim:]

                trh_res = tf.matmul(tf.tanh(tr + self.tr_combination_bias), ent_mat)
                _, head_ids = tf.nn.top_k(trh_res, k=self.n_entity)

                self.head_ids, self.tail_ids = head_ids, tail_ids

            else:
                hr = tf.matmul(tf.concat(1, [h, r]), self.hr_combination_matrix)
                hrt_res = (tf.matmul(tf.tanh(hr + self.hr_combination_bias), ent_mat))
                _, tail_ids = tf.nn.top_k(hrt_res, k=self.n_entity)

                tr = tf.matmul(tf.concat(1, [t, r]), self.tr_combination_matrix)
                trh_res = tf.matmul(tf.tanh(tr + self.tr_combination_bias), ent_mat)
                _, head_ids = tf.nn.top_k(trh_res, k=self.n_entity)

                self.head_ids, self.tail_ids = head_ids, tail_ids

            # triple score
            triple_score = tf.matmul(tf.tanh(hr + self.hr_combination_bias), ent_mat)

            self.triple_score_h = tf.reduce_sum(tf.multiply(tf.tanh(hr + self.hr_combination_bias), h), 1)
            self.triple_score_t = tf.reduce_sum(tf.multiply(tf.tanh(hr + self.hr_combination_bias), t), 1)

            return head_ids, tail_ids, hrt_res, trh_res, triple_score

        for p in active_children():
            p.terminate()

    def predict_proba(self, triples):
        triples = np.array(triples)
        test_input, _, _, _, _, triple_score = test_ops(self)

        scores_h, scores_t = self.sess.run([self.triple_score_h, self.triple_score_t],
                                           {test_input: triples})
        scores_pred = (scores_h + scores_t) / 2
        for p in active_children():
            p.terminate()
        return scores_pred.reshape((-1, 1))

    def predict(self, triples):
        return self.predict_proba(triples) > 0.5

    def close(self):
        self.sess.close()

    def learn_model(self, X, types=None, type_hierarchy=None, domains=None, ranges=None):

        train_triples = to_triples(X, order="sop")

        if self.add_types and types is not None:
            types = types.tocoo()
            n_entities = X[0].shape[0]
            rdf_type_id = len(X)
            type_triples = np.array([(types.row[i], types.col[i] + n_entities, rdf_type_id) for i in range(types.nnz)])
            train_triples = np.vstack((train_triples, type_triples))
            types = types.tocsr()
            self.n_entity += types.shape[1]
            self.n_relations += 1

        self.train_triples = train_triples
        self.train_hr_t = gen_hr_t(train_triples)
        self.train_tr_h = gen_tr_h(train_triples)

        self.create_model()

        train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight, \
        train_loss, train_op = train_ops(self, learning_rate=self.lr,
                                         optimizer_str=self.optimizer,
                                         regularizer_weight=self.loss_weight)

        test_input, test_head, test_tail, _, _, triple_score = test_ops(self)
        self.test_input = test_input

        self.sess = tf.Session()
        #tf.initialize_all_variables().run(session=self.sess)
        tf.global_variables_initializer().run(session=self.sess)

        self.saver = tf.train.Saver()

        iter_offset = 1

        total_inst = self.n_train

        # training data generator
        raw_training_data_queue = []
        training_data_queue = []
        data_generators = list()
        for i in range(self.n_generator):
            data_generators.append(Process(target=data_generator_func, args=(
                raw_training_data_queue, training_data_queue, self.train_tr_h, self.train_hr_t, self.n_entity,
                self.neg_weight)))
            data_generators[-1].start()

        for n_iter in range(iter_offset, self.max_iter + 1):
            start_time = timeit.default_timer()
            accu_loss = 0.
            accu_re_loss = 0.
            ninst = 0

            if self.verbose:
                print("initializing raw training data...")
            nbatches_count = 0
            for dat in self.raw_training_data(batch_size=self.batch):
                raw_training_data_queue.append(dat)
                nbatches_count += 1
            if self.verbose:
                print("raw training data initialized.")

            while nbatches_count > 0:
                nbatches_count -= 1

                hr_tlist, hr_tweight, tr_hlist, tr_hweight = training_data_queue[0]
                training_data_queue = training_data_queue[1:]

                l, rl, _ = self.sess.run(
                    [train_loss, self.regularizer_loss, train_op], {train_hrt_input: hr_tlist,
                                                                    train_hrt_weight: hr_tweight,
                                                                    train_trh_input: tr_hlist,
                                                                    train_trh_weight: tr_hweight})

                accu_loss += l
                accu_re_loss += rl
                ninst += len(hr_tlist) + len(tr_hlist)

            if self.verbose:
                print(
                    "iter %d avg loss %.5f, time %.3f" % (
                    n_iter, accu_loss / ninst, timeit.default_timer() - start_time))

            if self.save_per and n_iter and n_iter % self.save_per == 0:
                self.save_model(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "ckp/checkpoint-%d.ckp" % n_iter))

        for p in active_children():
            p.terminate()


def callback(m, with_eval=False):
    elapsed = timeit.default_timer() - m.epoch_start
    return True


class SKGEWrapper(ErrorDetector):
    def __init__(self, n_dim=150, n_batches=100, max_epochs=500, learning_rate=0.1, margin=0.2, rparam=0.1,
                 negative_examples=1, init="nunif", activation_function="sigmoid", model="hole", sample_mode="lcwa"):
        self.n_dim = n_dim
        self.n_batches = n_batches
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.margin = margin
        self.rparam = rparam
        self.negative_examples = negative_examples
        self.init = init
        self.activation_function = activation_function
        self.model = model
        self.sample_mode = sample_mode

    def learn_model(self, X, types=None, type_hierarchy=None, domains=None, ranges=None):
        self.X = X
        N = self.X[0].shape[1]
        self.sz = (N, N, len(self.X))
        if self.model == "hole":
            embedding_model = HolE(
                self.sz,
                self.n_dim,
                rparam=self.rparam,
                af=afs[self.activation_function],
                init=self.init
            )
        if self.model == "transe":
            embedding_model = TransE(self.sz, self.n_dim, init=self.init)

        if self.model == "rescal":
            embedding_model = RESCAL(self.sz, self.n_dim, init=self.init)

        xs = []
        for r, slice in enumerate(self.X):
            h = slice.row
            t = slice.col
            xs = xs + zip(h, t, [r] * len(h))

        ys = np.ones(len(xs))
        if self.sample_mode == 'corrupted':
            ti = type_index(xs)
            sampler = CorruptedSampler(self.negative_examples, xs, ti)
        elif self.sample_mode == 'random':
            sampler = RandomModeSampler(self.negative_examples, [0, 1], xs, self.sz)
        elif self.sample_mode == 'lcwa':
            sampler = LCWASampler(self.negative_examples, [0, 1, 2], xs, self.sz)

        self.trainer = PairwiseStochasticTrainer(
            embedding_model,
            nbatches=self.n_batches,
            max_epochs=self.max_epochs,
            post_epoch=[callback],
            learning_rate=self.learning_rate,
            margin=self.margin,
            samplef=sampler.sample
        )
        self.trainer.fit(xs, ys)
        del xs, ys

    def compute_scores(self):
        return self.predict_proba(self.true_triples)

    def detect_errors(self):
        pass

    def predict_proba(self, triples):
        sp = [s for s, o, p in triples]
        pp = [p for s, o, p in triples]
        op = [o for s, o, p in triples]
        if self.model == "hole":
            return np.sum(self.trainer.model.R[pp] * ccorr(self.trainer.model.E[sp], self.trainer.model.E[op]), axis=1)

        if self.model == "transe":
            score = self.trainer.model.E[sp] + self.trainer.model.R[pp] - self.trainer.model.E[op]
            return - np.sum(score ** 2, axis=1)

        if self.model == "rescal":
            return np.array([
                                np.dot(self.trainer.model.E[sp[i]],
                                       np.dot(self.trainer.model.W[pp[i]], self.trainer.model.E[op[i]]))
                                for i in range(len(sp))
                                ])

    def predict(self, triples):
        return (self.predict_proba(triples) > 0).astype(float)

    def prepare(self, mdl, p):
        if self.model == "hole":
            self.ER = ccorr(mdl.R[p], mdl.E)
        if self.model == "transe":
            self.ER = mdl.E + mdl.R[p]
        if self.model == "rescal":
            self.EW = np.mat(mdl.E) * np.mat(mdl.W[p])

    def scores_s(self, o, p):
        if self.model == "hole":
            return np.dot(self.trainer.model.E, self.ER[o])
        if self.model == "transe":
            return -np.sum(np.abs(self.ER - self.trainer.model.E[o]), axis=1)
        if self.model == "rescal":
            return -np.sum(np.abs(self.EW - self.trainer.model.E[o]), axis=1)

    def scores_o(self, s, p):
        if self.model == "hole":
            return np.dot(self.ER, self.trainer.model.E[s])
        if self.model == "transe":
            return -np.sum(np.abs(self.ER[s] - self.trainer.model.E), axis=1)
        if self.model == "rescal":
            return -np.sum(np.abs(self.EW[s] - self.trainer.model.E), axis=1)
