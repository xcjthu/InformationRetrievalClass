import json
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.sparse import csc_matrix
from matplotlib.font_manager import FontProperties
# plt.rcParams['font.sans-serif']=['SimHei']
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import os

plt.switch_backend('agg')

font=FontProperties(fname='STHeiti.ttc',size=4)

'''
font = {'family' : 'SimHei',
    # 'weight' : 'bold',
    'size'  : '8'}
'''
# plt.rc('font', **font)
from scipy.sparse.linalg import svds

class ScipySVD:
    def __init__(self, mat, K):
        self.mat = csc_matrix(mat, dtype=float)
        self.U, self.sigma, self.VT = svds(self.mat, k = K)
    
    def getTermMat(self):
        ttm = np.transpose(self.VT) @ np.diag(self.sigma)
        print(ttm.shape)
        return ttm


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
    
    def getTermMat(self, k):
        vt = self.VT[:k,]
        sigma = self.sigma[:k]
        allpoints = np.transpose(np.expand_dims(sigma, 1) * vt, axes=[1,0])
        return allpoints
    
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
    dtm_vectorizer = CountVectorizer()
    dtm = dtm_vectorizer.fit_transform(corpus)
    words = dtm_vectorizer.get_feature_names()
    word2id = {words[i]: i for i in range(len(words))}
    print(dtm.shape)
    return word2id, dtm
    '''
    word2id = {}
    for doc in corpus:
        for w in doc:
            if not w in word2id:
                word2id[w] = len(word2id)
    # mat = np.zeros((len(corpus), len(word2id)))
    mat = csc_matrix((len(corpus), len(word2id)), dtype = np.float)
    for docid, doc in enumerate(corpus):
        for w in doc:
            mat[docid, word2id[w]] += 1
    
    return word2id, mat
    '''

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

def save_fig(save_dir=None, title='fig.png', dpi=800, fig_dir='fig'):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(save_dir, fig_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=False)
        plt.close()
    return

def Task2(lines):
    word2id, dtm = getMat(lines)
    A = csc_matrix(dtm, dtype=float)

    k = 2

    sample_lines = np.random.permutation(lines)[:10]
    n_dtm_vectorizer = CountVectorizer()
    n_dtm_vectorizer.fit_transform(sample_lines)
    sample_w = n_dtm_vectorizer.get_feature_names()

    u, s, vh = svds(A, k=k)
    u = normalize(u, norm='l2', axis = 1)
    vh = normalize(vh, norm='l2', axis = 0)
    print(f"dtm:{A.shape}, U:{u.shape}, S:{s.shape}, VH:{vh.shape}")
    words = np.random.permutation(sample_w)[:30]
    print(words)
    x, y = [], []
    r = 1.04
    origin = [0], [0] # origin point
    for w in words:
        idx = word2id[w] # dtm_vectorizer.vocabulary_.get(w)
        rr = 1 + (np.random.rand() - 0.5)*0.1
        plt.annotate(w, (vh[0][idx]*rr, vh[1][idx]*rr),fontproperties=font)
    #     plt.annotate(f'w{ii}', (vh[0][idx], vh[1][idx]))
        x.append(vh[0][idx]*rr)
        y.append(vh[1][idx]*rr)
    # plt.quiver(*origin, x, y, color='g', sizes=(1,))
    plt.scatter(x, y, color='green', sizes=(1,))
    plt.scatter([[0]], [[0]], color='red', sizes=(5,))

    x, y = [], []
    r = 0.9
    for idx, d in enumerate(sample_lines):
        
        plt.annotate(f'd{idx}', (u[idx][0]*r, u[idx][1]*r),fontproperties=font)
        x.append(u[idx][0]*r)
        y.append(u[idx][1]*r)
        print(f'|d{idx} | {d}|')
    plt.scatter(x, y, color='blue', sizes=(1,))

    save_fig('.', 'fig2-2-2.png')

from mpl_toolkits.mplot3d import Axes3D
def Task2_3D(lines):
    word2id, dtm = getMat(lines)
    A = csc_matrix(dtm, dtype=float)

    k = 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sample_lines = np.random.permutation(lines)[:10]
    n_dtm_vectorizer = CountVectorizer()
    n_dtm_vectorizer.fit_transform(sample_lines)
    sample_w = n_dtm_vectorizer.get_feature_names()

    u, s, vh = svds(A, k=k)
    u = normalize(u, norm='l2', axis = 1)
    vh = normalize(vh, norm='l2', axis = 0)
    print(f"dtm:{A.shape}, U:{u.shape}, S:{s.shape}, VH:{vh.shape}")
    words = np.random.permutation(sample_w)[:30]
    print(words)
    x, y, z = [], [], []
    r = 1.04
    origin = [0], [0] # origin point
    # plt.hold()
    for w in words:
        idx = word2id[w]
        rr = 1 + (np.random.rand() - 0.5)*0.1
        ax.text(vh[0][idx]*rr, vh[1][idx]*rr, vh[2][idx]*rr, w,'x',fontproperties=font)
        x.append(vh[0][idx]*rr)
        y.append(vh[1][idx]*rr)
        z.append(vh[2][idx]*rr)
    # plt.quiver(*origin, x, y, color='g', sizes=(1,))
    ax.scatter(x, y, z, color='green', sizes=(1,))
    ax.scatter([[0]], [[0]], [[0]], color='red', sizes=(5,))
    # plt.hold()
    x, y, z = [], [], []
    r = 0.9
    for idx, d in enumerate(sample_lines):
        ax.text(u[idx][0]*r, u[idx][1]*r, u[idx][2]*r, f'd{idx}','x',fontproperties=font)
        x.append(u[idx][0]*r)
        y.append(u[idx][1]*r)
        z.append(u[idx][2]*r)
        print(f'|d{idx} | {d}|')
    ax.scatter(x, y, z, color='blue', sizes=(1,))

    save_fig('.', 'fig2-3-1.png')


def Task3(lines):
    word2id, dtm = getMat(lines)
    A = csc_matrix(dtm, dtype=float)
    id2word = {word2id[key]: key for key in word2id}

    u, s, vh = svds(A, k=3)
    ttm = np.transpose(vh) @ np.diag(s) @ vh
    print(ttm.shape)
    plt.pcolor(ttm[:100, :100])
    # plt.show()
    save_fig('.', '3-tt-100-3.png')

    for iii in ttm.reshape((-1,)).argsort()[-5:][::-1]:
        ii = np.unravel_index(iii, ttm.shape)[0]
        print(ii, ttm[ii][ii], id2word[ii])

def Task3_doc(lines):
    word2id, dtm = getMat(lines)
    A = csc_matrix(dtm, dtype=float)
    id2word = {word2id[key]: key for key in word2id}

    u, s, vh = svds(A, k=2)
    ddm = u @ np.diag(np.sqrt(s)) @ np.transpose(u)
    print(ddm.shape)
    plt.pcolor(ddm[:100, :100])
    save_fig('.', '3-dd-100-2.png')

    for iii in ddm.reshape((-1,)).argsort()[-5:][::-1]:
        ii = np.unravel_index(iii, ddm.shape)[0]
        print(ii, ddm[ii][ii], lines[ii])

def getWordVec():
    sizes = [100000, 400000, 3000000, 5000000]
    fin = open('../../data/rmrb_cut.txt', 'r')
    corpus = [' '.join(json.loads(fin.readline())) for i in range(max(sizes))]

    for size in sizes:
        print(size)
        cor = corpus[:size]
        word2id, mat = getMat(cor)
        id2word = {word2id[key]: key for key in word2id}
        # svd = SVDClass(mat)
        svd = ScipySVD(mat, 100)
        
        wordvec = svd.getTermMat()
        fout = open('model/lsi%d.txt' % size, 'w')
        print("%d %d" % (len(id2word), 100), file=fout)
        for i in range(len(id2word)):
            print(id2word[i], end=' ', file=fout)
            for m in wordvec[i]:
                print(m, end=' ', file=fout)
            print(file=fout)
        fout.close()


if __name__ == '__main__':
    '''
    corpus = readCorpus('../data/sogou.txt', 2000)
    # analyseK(corpus)
    plotPoint(corpus)
    '''
    fin = open('../../data/rmrb_cut.txt', 'r')
    corpus = [' '.join(json.loads(fin.readline())) for i in range(5000)]
    # Task2(corpus)
    # Task2_3D(corpus)
    # Task3(corpus)
    Task3_doc(corpus)
    # getWordVec() 




