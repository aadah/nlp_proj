import sys
import os.path
import config
import collections
import numpy as np
import scipy.spatial as sps
from scipy import stats
from sklearn.cluster import DBSCAN

np.random.seed(1)

class KNN:
    def __init__(self, k, rep=''):
        self.k = k
        self.rep = rep
        if rep == '':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_NPY)
        elif rep == 'subtract':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_SUBTRACT_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_SUBTRACT_NPY)
        elif rep == 'autoencode':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_AUTOENCODE_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_AUTOENCODE_NPY)
            '''
            N,D = self.XY_test.shape
            newXY = np.copy(self.XY_test)
            self.XY_test[:,:D/2-1] = newXY[:,D/2-1:-2]
            self.XY_test[:,D/2-1:-2] = newXY[:,:D/2-1]
            self.XY_test[:,-2] = newXY[:,-1]
            self.XY_test[:,-1] = newXY[:,-2]
            '''
        elif rep == 'pca':
            self.XY_train = np.load(config.PCA_DATA_TRAIN_NPY)
            self.XY_test = np.load(config.PCA_DATA_TEST_NPY)
        elif rep == 'pca_subtract':
            self.XY_train = np.load(config.PCA_DATA_TRAIN_SUBTRACT_NPY)
            self.XY_test = np.load(config.PCA_DATA_TEST_SUBTRACT_NPY)
        elif rep == 'pca_autoencode':
            self.XY_train = np.load(config.PCA_DATA_TRAIN_AUTOENCODE_NPY)
            self.XY_test = np.load(config.PCA_DATA_TEST_AUTOENCODE_NPY)
        
    def train(self):
        print 'Training . . .'
        N, D = self.XY_train.shape
        X = np.empty((2*N, (D-2)/2))
        X[:N,:] = self.XY_train[:,:(D-2)/2]
        X[N:,:] = self.XY_train[:,(D-2)/2:-2]
        Y = np.empty((2*N,1))
        Y[:N,0] = self.XY_train[:,-2]
        Y[N:,0] = self.XY_train[:,-1]
        relations2labels = {}
        for i in xrange(2*N):
            relations2labels[self._vec2str(X[i])] = int(Y[i][0])
        #print '# relation labels:', len(set(relations2labels.values()))
        #print '# unique relations:', len(relations2labels.keys())
        self.relations = np.empty((len(relations2labels),(D-2)/2))
        self.labels = np.empty((len(relations2labels),1))
        i = 0
        for x_str in relations2labels:
            self.relations[i] = self._str2vec(x_str)
            self.labels[i] = relations2labels[x_str]
            i+=1
        #self.relation_indices = {self.relations[i]:i for i in xrange(len(self.relations))}
        sys.stdout.flush()

    def test(self):
        N, D = self.XY_test.shape
        Y = self.XY_test[:,-2:]
        pred = self.classify()
        TP = FP = FN = TN = 0
        res = np.empty((N,))
        for i in xrange(N):
            l1 = pred[i]
            l2 = pred[N+i]
            #print 'pred',l1,l2
            #print 'Y',Y[i,0],Y[i,1]
            if l1 == l2 and Y[i,0] == Y[i,1]:
                TP += 1
            elif l1 == l2 and Y[i,0] != Y[i,1]:
                FP += 1
            elif l1 != l2 and Y[i,0] == Y[i,1]:
                FN += 1
            elif l1 != l2 and Y[i,0] != Y[i,1]:
                TN += 1
            res[i] = 1-np.bitwise_xor(int(np.sign(abs(l1-l2))),int(np.sign(abs(Y[i,0]-Y[i,1]))))
        print 'TP:',TP
        print 'FP:',FP
        print 'FN:',FN
        print 'TN:',TN
        print 'Accuracy:',float(np.sum(res))/N
        sys.stdout.flush()

    def classify(self):
        self.build_top_indices()
        N_test, _ = self.top_indices.shape
        pred = np.empty((N_test,))
        for i in xrange(N_test):
            top_indices = self.top_indices[i,:k]
            #print top_indices
            #print [self.labels[int(j)] for j in top_indices]
            pred[i] = stats.mode([self.labels[int(j)] for j in top_indices])[0][0]
        return pred
            
    def build_dist_matrix(self):        
        fname = 'distance_matrix_%s.npy' %self.rep
        if os.path.isfile(fname):
            #print 'loading %s . . .' %fname
            self.distances = np.load(fname)
        else:
            #print 'building distance matrix . . .'
            N_relations, _ = self.relations.shape
            #print self.relations.shape
            # where N_relations is the number of unique relations in the training set
            N_test, D = self.XY_test.shape
            self.distances = np.empty((2*N_test,N_relations))
            #print self.distances.shape
            #print 'test relations',N_test
            for i in xrange(N_test):
                #print 'row',i
                # stack
                x1 = self.XY_test[i,:(D-2)/2]
                x2 = self.XY_test[i,(D-2)/2:-2]
                self.distances[i] = np.array(
                    [self._distance(x1,self.relations[j]) for j in xrange(N_relations)])
                self.distances[N_test+i] = np.array(
                    [self._distance(x2,self.relations[j]) for j in xrange(N_relations)])
            np.save(fname, self.distances)
        #print 'distances: ',self.distances.shape
        sys.stdout.flush()
            
    def build_top_indices(self):
        if os.path.isfile('top_indices_%s.npy' %self.rep):
            #print 'loading top_indices . . .'
            self.top_indices = np.load('top_indices_%s.npy' %self.rep)
            sys.stdout.flush()
            return
        #print 'creating top_indices . . .'
        self.build_dist_matrix()
        N_test, N_relations = self.distances.shape
        # where N_relations is the number of unique relations in the training set
        self.top_indices = np.empty(self.distances.shape)
        for j in xrange(N_test):
            row = self.distances[j]
            tagged_row = [(row[i], i) for i in xrange(N_relations)]
            tagged_row.sort(key=lambda d: d[0]) # sort by distance
            self.top_indices[j] = np.array([d[1] for d in tagged_row])
        #print 'top_indices',self.top_indices.shape
        np.save('top_indices_%s.npy' %self.rep, self.top_indices)
        sys.stdout.flush()
        
    def _vec2str(self, arr):
        return str(list(arr))
    
    def _str2vec(self, str):
        #print np.array(eval(str))
        return np.array(eval(str))    

    def _distance(self, u, v):
        #print sps.distance.euclidean(u,v)
        return sps.distance.euclidean(u,v)
    

if __name__=='__main__':
    for k in [1,2,5,10,15,30,45,60]:
    #for k in [10, 15, 20]:
        for rep in ['','pca','subtract','pca_subtract','autoencode','pca_autoencode']:
        #for rep in ['autoencode']:
            print 'k =',k
            print 'rep:',rep
            knn = KNN(k,rep)
            knn.train()
            knn.test()
