import json
import re
import numpy as np
import matplotlib.pyplot as plt
import random
# plt.rcParams['font.sans-serif']=['SimHei']
font = {'family' : 'SimHei',
    # 'weight' : 'bold',
    'size'  : '8'}
plt.rc('font', **font)

class SVDClass:
    def __init__(self, mat):
        self.mat = mat
        self.U, self.sigma, self.VT = np.linalg.svd(mat)
        print("shape: mat", mat.shape)

    def getAK(self, k):
        uk = self.U[:,:k]
        sigma = self.sigma[:k]
        vt = self.VT[:k,]
    
        ak = np.expand_dims(sigma, 0)
        ak = ak * uk # np.matmul(uk, sigma)
        ak = np.matmul(ak, vt)
        return ak, np.linalg.norm(self.mat - ak)
    
    def getTermPoint(self, k, pindex):
        vt = self.VT[:k,]
        sigma = self.sigma[:k]
        allpoints = np.transpose(np.expand_dims(sigma, 1) * vt, axes=[1,0])
        # print(allpoints.shape)
        points = [allpoints[index] for index in pindex]
        return points / (np.expand_dims(np.linalg.norm(points, axis=1), 1) + np.random.rand(len(pindex), 1))
    
    def getDocPoint(self, k, pindex):
        uk = self.U[:,:k]
        sigma = self.sigma[:k]
        allpoints = np.expand_dims(sigma, 0) * uk

        points = [allpoints[index] for index in pindex]
        return points / (np.expand_dims(np.linalg.norm(points, axis=1), 1) + np.random.rand(len(pindex), 1))

def readCorpus(path, num = 1000):
    fin = open(path, 'r')
    corpus = []
    for line in fin:
        line = re.sub('/[a-z]+[ \n]', ' ', line)
        corpus.append(line.strip().split())
        if len(corpus) >= num:
            break
    return corpus

def getMat(corpus):
    word2id = {}
    for doc in corpus:
        for w in doc:
            if not w in word2id:
                word2id[w] = len(word2id)
    mat = np.zeros((len(corpus), len(word2id)))
    for docid, doc in enumerate(corpus):
        for w in doc:
            mat[docid, word2id[w]] += 1
    
    return word2id, mat

def analyseK(corpus):
    print('corpus size:', len(corpus))
    word2id, mat = getMat(corpus)

    svd = SVDClass(mat)

    klist = [2, 4, 8, 16, 32, 50, 100, 200, 300, 400, 500, 700, 900, 1100, 1300, 1600, 2000]
    error = []
    for k in klist:
        a, er = svd.getAK(k)
        error.append(er)
    plt.plot(klist, error, marker='o')
    plt.xlabel('k') 
    plt.ylabel('l2-error')
    plt.show()
    print(klist)
    print(error)

def plotPoint(corpus):
    print('corpus size:', len(corpus))
    word2id, mat = getMat(corpus)
    id2word = {word2id[key]: key for key in word2id}

    svd = SVDClass(mat)

    docIndex = random.sample(range(len(corpus)), 15)
    wordIndex = set()
    for docInd in docIndex:
        print(''.join(corpus[docInd]))
        for w in corpus[docInd]:
            wordIndex.add(word2id[w])
    wordIndex = list(wordIndex)

    docPoints = svd.getDocPoint(2, docIndex)
    termPoints = svd.getTermPoint(2, wordIndex[:30])

    plt.scatter([p[0] for p in docPoints], [p[1] for p in docPoints], marker = 'x', color = 'blue', s = 40 ,label = 'doc')
    plt.scatter([p[0] for p in termPoints], [p[1] for p in termPoints], marker = 'o', color = 'green', s = 40 ,label = 'term')
    for i in range(len(termPoints)):
        plt.annotate(id2word[wordIndex[i]], xy = (termPoints[i][0], termPoints[i][1]), xytext = (termPoints[i][0], termPoints[i][1]))
    plt.legend(loc = 'best')
    plt.show()


if __name__ == '__main__':
    corpus = readCorpus('../data/sogou.txt', 2000)
    # analyseK(corpus)
    plotPoint(corpus)
        




