from os.path import abspath, isfile
from tqdm import tqdm

from csv import reader as csv_reader
from sklearn.feature_extraction.text import CountVectorizer

from json import dump as json_dump, load as json_load
from jsonpickle import encode as jp_encode, decode as jp_decode

from server.text_processor import TextProcessor


class InvertedIndex:
    def __init__(self, fpath, dump_fpath):
        self.text_processor = TextProcessor()

        self.queries = []
        self.documents = []
        self.original_documents = []
        self.is_duplicates = []

        self.vectorizer = CountVectorizer()
        self.build_index(fpath)

        self.dump(dump_fpath)

    def build_index(self, fpath):
        with open(abspath(fpath), 'r', encoding='utf-8') as file:
            table = csv_reader(file)

            for row in tqdm(list(table)):
                if row[0] == '':
                    continue
                # TODO: сделать полный индекс
                if row[0] == '10000':
                    break

                self.queries.append(
                    self.text_processor.process(row[1])
                )
                self.documents.append(
                    self.text_processor.process(row[2])
                )

                self.original_documents.append(row[2])
                self.is_duplicates.append(row[3])

        self.vectorizer.fit_transform(self.queries)

    def dump(self, dump_fpath):
        json_encoded = jp_encode(self)

        with open(dump_fpath, 'w', encoding='utf-8') as file:
            json_dump(json_encoded, file, ensure_ascii=False, indent=4)

    @staticmethod
    def restore(dump_fpath):
        with open(dump_fpath, "r", encoding='utf-8') as file:
            idx_dump = json_load(file)
        return jp_decode(idx_dump)

    @staticmethod
    def from_dump_or_build(dump_fpath, corpora_fpath):
        if isfile(dump_fpath):
            try:
                return InvertedIndex.restore(dump_fpath)
            except Exception:

                return InvertedIndex(corpora_fpath, dump_fpath)
        else:
            return InvertedIndex(corpora_fpath, dump_fpath)

