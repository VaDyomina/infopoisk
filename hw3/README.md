## Алгоритм запуска:
#### 1) `pip3 install virtualenv`
#### 2) `python3 -m virtualenv venv --distribute`
#### 3) `source venv/bin/activate`
#### 4) In virtualenv: `pip3 install -r requirements.txt`
#### 4) In virtualenv: `python3 -m nltk.downloader stopwords`


#### Для того, чтобы сделать запрос: `python3 searcher.py --query="соленая вода"` 
#### Для вызова помощи: `python3 searcher.py --help` 


## Пример вывода:

Вот что нашлось fasttext`ом по запросу "соленая вода" за 1.19218111038208 сек:

1) ('0.8905', 'какая рыба выживет в соленой воде')
2) ('0.8619', 'почему тафтинги из соленой воды импортируются в португальском')
3) ('0.8463', 'почему в Японии импортируется соленая вода')
4) ('0.8455', 'почему тафтинка из соленой воды импортируется в италию')
5) ('0.8154', 'вода в бутылках лучше для вас, чем водопроводная вода')
6) ('0.8071', 'как вода отличается от дистиллированной воды')
7) ('0.8033', 'как дистиллированная вода и очищенная вода одинаковы')
8) ('0.7608', 'мы можем использовать дождевую воду для заполнения грунтовых вод')
9) ('0.7598', 'как я растворяю сахар в холодной воде')
10) ('0.7447', 'почему важно промыть соленой водой после удаления зубов мудрости')