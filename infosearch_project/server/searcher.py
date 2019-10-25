from server.rankers.tf_idf import TfIdf
from server.rankers.bm25 import Bm25
from server.rankers.fasttext import Fasttext
from server.rankers.elmo import Elmo


class Searcher:
    def __init__(self, inverted_index):
        self.tf_idf = TfIdf(inverted_index)
        self.bm25 = Bm25(inverted_index)
        self.fasttext = Fasttext(inverted_index)
        self.elmo = Elmo(inverted_index)

    def search(self, ranker, query):
        if ranker == 'tf_idf':
            return self.tf_idf.search(query)
        elif ranker == 'bm25':
            return self.bm25.search(query)
        elif ranker == 'fasttext':
            return self.fasttext.search(query)
        elif ranker == 'elmo':
            return self.elmo.search(query)
        else:
            raise LookupError()
