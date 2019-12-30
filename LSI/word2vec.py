import gensim
from gensim.models import Word2Vec
import json

sizes = [100000, 400000, 3000000, 5000000]
corpus_path = '../../data/rmrb_cut.txt'
size = max(sizes)
corpus = []
fin = open(corpus_path, 'r')
for line in fin:
    corpus.append(json.loads(line))
    if len(corpus) >= size:
        break

for size in sizes:
    print(size)
    # corpus = [json.loads(line) for line in open(corpus_path, 'r')][:100]
    word2vec = Word2Vec(corpus[:size], size = 100, window = 5, min_count = 5, workers = 10)
    word2vec.wv.save_word2vec_format('model/word2vec%d.txt' % size, binary=False)

