# -*- coding: utf-8 -*-

from time import time
from os import environ
from argparse import ArgumentParser

from elmo import Elmo
from fasttext import Fasttext
from inverted_index import InvertedIndex

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    arg_parser = ArgumentParser(description='Search with fasttext and elmo')
    arg_parser.add_argument('--query', type=str, help='Query you want to search in quora')
    query = arg_parser.parse_args().query

    inverted_index = InvertedIndex.from_dump_or_build(
        './dumps/inverted_index.json',
        './quora_question_pairs_rus.csv'
    )

    print("\nНачну поиск c fasttext'ом...\n")

    # ------------------ Fasttext part start --------------------------- #

    fasttext_searcher = Fasttext(inverted_index)

    start_fasttext_time = time()
    fasttext_search_result = fasttext_searcher.search(query)
    fasttext_elapsed_time = time() - start_fasttext_time

    print(f'Вот что нашлось fasttext`ом по запросу "{query}" за {fasttext_elapsed_time} сек:\n')
    for index, result in enumerate(fasttext_search_result):
        print(f'{index + 1}) {result}')

    # ------------------ Fasttext part end --------------------------- #

    input("\nНажмите Enter чтобы поискать ещё и с Elmo...\n")

    # ------------------ Elmo part start --------------------------- #

    elmo_searcher = Elmo(inverted_index)

    start_elmo_time = time()
    elmo_search_result = elmo_searcher.search(query)
    elmo_elapsed_time = time() - start_elmo_time

    print(f'Вот что нашлось elmo`м по запросу "{query}" за {elmo_elapsed_time} сек:')
    for index, result in enumerate(elmo_search_result):
        print(f'{index + 1}) {result}')

    # ------------------ Elmo part end --------------------------- #
