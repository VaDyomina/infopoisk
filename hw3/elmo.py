# -*- coding: utf-8 -*-

from os import getcwd
from os.path import isfile, abspath, join

from numpy import mean, zeros
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from pickle import dump as pickle_dump, load as pickle_load

from external.simple_elmo.elmo_helpers import get_elmo_vectors, load_elmo_embeddings


class Elmo:
    def __init__(self, inverted_index):
        self.batcher, self.sentence_character_ids, self.elmo_sentence_input = load_elmo_embeddings(
            abspath(join(getcwd(), 'external/simple_elmo/elmo'))
        )

        self.inverted_index = inverted_index
        self.splitted_documents = [document.split(' ') for document in self.inverted_index.documents]
        self.storage = ElmoStorage.from_dump_or_create(
            abspath(
                join(getcwd(), 'dumps/simple_elmo.pickle')
            ),
            self.init_elmo_vectors
        )

    def init_elmo_vectors(self):
        elmo_vectors = zeros((len(self.splitted_documents), 1024))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 100
            docs_limiter = 200

            for i in range(0, len(self.splitted_documents[:docs_limiter]), 100):
                elmo_vectors[i:i + batch_size] = mean(
                    get_elmo_vectors(sess, self.splitted_documents[i:i + batch_size], self.batcher,
                                     self.sentence_character_ids, self.elmo_sentence_input), axis=1
                )

        return elmo_vectors

    def search(self, query, size=10):
        tf.reset_default_graph()

        preprocessed_query = self.inverted_index.text_processor.process(query).split(' ')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            query_elmo_vector = mean(get_elmo_vectors(session, [preprocessed_query], self.batcher,
                                                      self.sentence_character_ids, self.elmo_sentence_input), axis=1)[0]
            print(query_elmo_vector)
            print(self.storage.elmo_vectors[1])

        similarity = cosine_similarity(
            self.storage.elmo_vectors, query_elmo_vector.reshape(1, -1)
        ).reshape(self.storage.elmo_vectors.shape[0])

        return [(metric, self.inverted_index.documents[document]) for document, metric in sorted(
            list(enumerate(similarity)), key=lambda d: d[1], reverse=True
        )][:size]


class ElmoStorage:
    def __init__(self, dump_fpath, elmo_vectors):
        self.elmo_vectors = elmo_vectors
        self.dump(dump_fpath)

    def dump(self, dump_fpath):
        with open(dump_fpath, 'wb') as file:
            pickle_dump(self, file)

    @staticmethod
    def restore(dump_fpath):
        with open(dump_fpath, "rb") as file:
            dump = pickle_load(file)
        return dump

    @staticmethod
    def from_dump_or_create(dump_fpath, init_elmo_vectors):
        if isfile(dump_fpath):
            try:
                return ElmoStorage.restore(dump_fpath)
            except Exception:
                return ElmoStorage(dump_fpath, init_elmo_vectors())
        else:
            return ElmoStorage(dump_fpath, init_elmo_vectors())
