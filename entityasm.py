import re
from collections import defaultdict as ddict
import dill
import scipy.sparse as sp
import unidecode
from rdflib import URIRef

from util import short_str, dameraulevenshtein, get_deletes_list


class EntityDisamb:
    def __init__(self, path, ents_dict):
        self.ents_dict = ents_dict
        self.inv_ents_dict = {k: v for v, k in self.ents_dict.items()}
        self.sim_sets = ddict(lambda: set())
        rows, cols = [], []
        for line in open(path, "rb"):
            triple = line.split(" ")
            if len(triple) == 4 and triple[0] and triple[2]:
                s, o = triple[0][1:-1], triple[2][1:-1]
                s = URIRef(s)
                o = URIRef(o)

                s_id, o_id = None, None
                if s in self.ents_dict:
                    s_id = ents_dict[s]
                if o in self.ents_dict:
                    o_id = ents_dict[o]

                if s_id and o_id:
                    rows += [s_id, o_id]
                    cols += [o_id, s_id]
                else:
                    if not s_id and o_id:
                        self.sim_sets[s].add(o_id)
                    if not o_id and s_id:
                        self.sim_sets[o].add(s_id)

        for k, v in self.sim_sets.items():
            if len(v):
                for i in v:
                    for j in v:
                        if j != i:
                            rows.append(i)
                            cols.append(j)
        self.A = sp.coo_matrix(([True] * len(rows), (rows, cols)), shape=(len(ents_dict), len(ents_dict)), dtype=bool)
        self.A = self.A.tocsr()

    def __getitem__(self, key):
        return self.A.__getitem__(key)

    def suggestions(self, ent):
        return self.A[ent].indices

    def save(self, path):
        dill.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return dill.load(open(path, "rb"))


class EntityASM:
    def __init__(self, max_edit_distance=1, k=10, verbose=True):
        self.dictionary = {}
        self.hash_deletes = ddict(lambda: [])
        self.similar_tokens = ddict(lambda: [])
        self.max_edit_distance = max_edit_distance
        self.k = k
        self.verbose = verbose
        self.longest_word_length = 0
        self.ent_word_counts = {}

    def create_dictionary_entry(self, w, entity=[]):
        # print(w)
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)

        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0] + 1, self.dictionary[w][1] + entity)
        else:
            self.dictionary[w] = (1, entity)
            self.longest_word_length = max(self.longest_word_length, len(w))

            self.hash_deletes[w].append(w)
            deletes = get_deletes_list(w, self.max_edit_distance)
            for item in deletes:
                self.hash_deletes[item].append(w)

    def create_dictionary(self, ents_dict):
        if isinstance(ents_dict.keys()[0], int):
            self.ents_dict = ents_dict
        else:
            self.ents_dict = {k: v for v, k in ents_dict.items()}
        total_word_count = 0
        unique_word_count = 0
        print("Creating dictionary...")
        for id, ent in self.ents_dict.items():
            # separate by words by non-alphabetical characters
            line = short_str(ent)
            line = line.decode("utf-8")
            line = unidecode.unidecode(line)
            words = self.get_words(line)
            self.ent_word_counts[id] = len(words)
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word, [id]):
                    unique_word_count += 1

        for k, tokens in self.hash_deletes.items():
            for t in tokens:
                self.similar_tokens[t].extend(tokens)

        for k, v in self.similar_tokens.items():
            self.similar_tokens[k] = set(v)
        del self.hash_deletes

        if self.verbose:
            print("total words processed: %i" % total_word_count)
            print("total unique words in corpus: %i" % unique_word_count)
            print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
            print("  edit distance for deletions: %i" % self.max_edit_distance)
            print("  length of longest word in corpus: %i" % self.longest_word_length)

    def get_words(self, string):
        string = re.sub("_?\(.*\)", "", string.lower())
        words = re.findall('[a-z]+', string)  # [1:]
        return words

    def get_similar_entities(self, entity, id=True, silent=True, match_all_words=False):
        string = short_str(entity)
        return self.get_suggestions(string, id=id, silent=silent, match_all_words=match_all_words)

    def get_suggestions(self, string, id=True, silent=True, match_all_words=False):
        query_words = self.get_words(string)
        all_sugestions = []
        len_query = len(query_words)
        for word in query_words:
            all_sugestions += self.get_word_suggestions(word, silent=silent)

        results = {}
        for word, (freq, dist, ent_ids, w) in all_sugestions:
            for ent_id in ent_ids:
                ent = ent_id if id else self.ents_dict[ent_id]
                if ent not in results:
                    results[ent] = (set([w]), dist, self.ent_word_counts[ent_id])
                else:
                    results[ent][0].add(w)
                    results[ent] = (results[ent][0], results[ent][1] + dist, results[ent][2])

        if match_all_words:
            results = {ent: (matches, dist, nwords) for ent, (matches, dist, nwords) in results.items() if
                       len(matches) == len_query}

        results = sorted(results.items(), key=lambda ent, matches_dist_nwords: (
            -len(matches_dist_nwords[0]), matches_dist_nwords[1] + 1.5 * float(len_query - len(matches_dist_nwords[0])) / len_query))
        return results[:self.k]

    def get_word_suggestions(self, string, silent=True):
        '''return list of suggested corrections for potentially incorrectly
           spelled word'''
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance (%d - %d) > %d" %
                      (len(string), self.longest_word_length, self.max_edit_distance))
                return []

            suggest_dict = {}

            # process queue item
            if (string in self.dictionary) and (string not in suggest_dict):

                suggest_dict[string] = (
                    self.dictionary[string][0], len(string) - len(string), self.dictionary[string][1], string)
                # the suggested corrections for string as stored in
                # dictionary (whether or not string itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.similar_tokens[string]:
                    if (sc_item not in suggest_dict):
                        # compute edit distance
                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)
                        suggest_dict[sc_item] = (
                            self.dictionary[sc_item][0], item_dist, self.dictionary[sc_item][1], string)

            # queue is now empty: convert suggestions in dictionary to
            # list for output
            if not silent and self.verbose != 0:
                print("number of possible corrections: %i" % len(suggest_dict))
                print("  edit distance for deletions: %i" % self.max_edit_distance)

            # output option 1
            # sort results by ascending order of edit distance and descending
            # order of frequency
            #     and return list of suggested word corrections only:
            # return sorted(suggest_dict, key = lambda x:
            #               (suggest_dict[x][1], -suggest_dict[x][0]))

            # output option 2
            # return list of suggestions with (correction,
            #                                  (frequency in corpus, edit distance)):
            as_list = list(suggest_dict.items())
            outlist = sorted(as_list, key=lambda term, freq_dist_ents_w: (freq_dist_ents_w[1], -freq_dist_ents_w[0]))

            return outlist

        def save(self, path):
            dill.dump(self, open(path, "wb"))

        @staticmethod
        def load(path):
            return dill.load(open(path, "rb"))
            print(" ")
