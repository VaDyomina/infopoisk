# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from pymystem3 import Mystem
mystem = Mystem()


class TextProcessor:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'[а-яА-ЯёЁ]+')
        self.stop_words = stopwords.words('russian')

    def process(self, text):
        tokenized = self.tokenizer.tokenize(text)
        lemmatized = [mystem.lemmatize(token)[0] for token in tokenized]
        without_stopwords = [word for word in lemmatized if word not in self.stop_words]

        return " ".join(without_stopwords)
