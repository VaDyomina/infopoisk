from math import log
from statistics import mean
from pandas import DataFrame


class Bm25:
    def __init__(self, inverted_index):
        self.default_k = 2.0
        self.default_b = 0.75
        self.inverted_index = inverted_index

        self.inverted_index_df = DataFrame(
            self.inverted_index.vectorizer.transform(inverted_index.queries).A,
            columns=self.inverted_index.vectorizer.get_feature_names(),
            index=inverted_index.documents
        ).transpose()

    def search(self, query, k=None, b=None, size=10):
        if k is None:
            k = self.default_k
        if b is None:
            b = self.default_k

        normalized_query = self.inverted_index.text_processor.process(query).split()

        N = self.inverted_index_df.shape[1]
        avgdl = mean([len(document) for document in self.inverted_index.documents])

        result = []

        for word in normalized_query:
            nq = self.inverted_index_df.sum(axis=1)[word] if word in self.inverted_index_df.transpose() else 0

            for index, document in enumerate(self.inverted_index.documents):
                ld = len(document.split())
                tf = self.inverted_index_df[document][word] if word in self.inverted_index_df[document] else 0

                idf = log((N - nq + 0.5) / (nq + 0.5))
                score = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (ld / avgdl)))

                if score != 0:
                    result.append((score, self.inverted_index.original_documents[index]))

        return sorted(result, key=lambda score_doc: score_doc[0], reverse=True)[:size]
