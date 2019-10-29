# -*- coding: utf-8 -*-

from os import getcwd
from os.path import join

from numpy import zeros, mean
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


class Fasttext:
    def __init__(self, inverted_index):
        self.bias = 0.3
        self.inverted_index = inverted_index

        self.model = KeyedVectors.load(
            join(getcwd(), 'external', 'fasttext', 'model.model')
        )

        self.fasttext_vectors = zeros(
            (len(self.inverted_index.documents), self.model.vector_size)
        )

        for index, document in enumerate(self.inverted_index.documents):
            self.fasttext_vectors[index] = self.get_vector(document.split(' '))

    def get_vector(self, document):
        token_vectors = zeros(
            (len(document), self.model.vector_size)
        )

        for index, token in enumerate(document):
            if token in self.model.vocab:
                token_vectors[index] = self.model.wv[token]

        return mean(token_vectors, axis=0)

    def search(self, query, size=10):
        query_vector = self.get_vector(
            self.inverted_index.text_processor.process(query).split(' ')
        )

        similarity = cosine_similarity(self.fasttext_vectors,
                                       query_vector.reshape(1, -1)).reshape(self.fasttext_vectors.shape[0])

        results = [(str(float('{:.4f}'.format(simil))), self.inverted_index.original_documents[document])
                   for document, simil in sorted(list(enumerate(similarity)), key=lambda d: d[1], reverse=True)][:size]

        return list(filter(lambda res: float(res[0]) > self.bias, results))
