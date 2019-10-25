from numpy import argsort


class TfIdf:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def search(self, query, size=10):
        search_matrix = self.inverted_index.vectorizer.transform([query] + self.inverted_index.documents)
        scores = argsort((search_matrix[0, :] * search_matrix[1:, :].T).A[0])

        return [(str(score), self.inverted_index.original_documents[score]) for score in scores[::-1]][:size]
