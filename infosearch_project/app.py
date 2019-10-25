# -*- coding: utf-8 -*-

from os import environ
from time import time

from json import dumps as json_dumps
from flask import Flask, request, render_template

from server.inverted_index import InvertedIndex
from server.searcher import Searcher

inverted_index = None
searcher = None

app = Flask(
    __name__,
    static_url_path='',
    static_folder='client/static',
    template_folder='client/templates'
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=['POST'])
def search():
    query = request.form['query']
    ranker = request.form['ranker']

    if query == "":
        return json_dumps({'error': 'Запрос-то нужно ввести!'}, ensure_ascii=False)

    try:
        start_time = time()
        search_result = searcher.search(ranker, query)
        elapsed_time = time() - start_time

        return json_dumps({
                'query': query,
                'ranker': ranker,
                'elapsed_time': float('{:.3f}'.format(elapsed_time)),
                'search_result': search_result
            }, ensure_ascii=False)
    except LookupError:
        return json_dumps({'error': 'Метод пока реализован!'}, ensure_ascii=False)


if __name__ == '__main__':
    inverted_index = InvertedIndex.from_dump_or_build(
        './dumps/inverted_index.json',
        './corpora/quora_question_pairs_rus.csv'
    )

    searcher = Searcher(inverted_index)

    port = int(environ.get("PORT", 3000))
    app.run(port=port, host='0.0.0.0', debug=True)
