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
        self.nodes = {}
        self.clusters = collections.defaultdict(set)
        if rep == '':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_NPY)
        elif rep == 'subtract':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_SUBTRACT_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_SUBTRACT_NPY)
        elif rep == 'autoencode':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_AUTOENCODE_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_AUTOENCODE_NPY)
    
        
    def build_entity_pair_list(self):
        print 'Building entity pair list . . .'
        keys = self.nodes.keys()
        n = len(keys)
        D = (self.XY_train.shape[1]-1)/2
        self.entity_pairs = np.empty((n,D))
        self.entity_pair_labels = []
        for i in xrange(n):
            self.entity_pairs[i] = self._str2vec(keys[i])
            self.entity_pair_labels.append(self.clusters[keys[i]])

    '''
    def train(self):
        print 'Training . . .'
        N, D = self.XY_train.shape
        Y = self.XY_train[:,-1]
        X1 = self.XY_train[:,:(D-1)/2]
        X2 = self.XY_train[:,(D-1)/2:-1]
        self.nodes = {}
        print 'Clustering . . .'
        
        for i in xrange(N):
            x1_str = self._vec2str(X1[i])
            if x1_str in self.nodes:
                x1 = self.nodes[x1_str]
            else:
                x1 = NodeSet(X1[i])
                self.nodes[x1_str] = x1
            x2_str = self._vec2str(X2[i])
            if x2_str in self.nodes:
                x2 = self.nodes[x2_str]
            else:
                x2 = NodeSet(X2[i])
                self.nodes[x2_str] = x2
            if Y[i] == 1:
                union(x1,x2)
        labels = {}
        cluster_index = collections.defaultdict(list)
        for node in self.nodes.values():
            node_root = self._vec2str(find(node).arr)
            if node_root not in labels:
                labels[node_root] = len(labels)
            self.clusters[self._vec2str(node.arr)] = int(labels[node_root])
            cluster_index[labels[node_root]].append(node)
        print '# unique relations:', len(self.nodes)
        print '# clusters:',len(set(self.clusters.values()))
        self.build_entity_pair_list()
    '''
    
    def train(self):
        print 'Training . . .'
        N, D = self.XY_train.shape
        X = self.XY_train[:,:-1]
        Y = self.XY_train[:,-1]
        for i in xrange(N):
            self.clusters[self._vec2str(X[i])] = Y[i]
        print '# clusters:', len(set(self.clusters.values()))
        self.entity_pairs = self.XY_train[:,:-1]
        self.entity_pair_labels = self.XY_train[:,-1]
        sys.stdout.flush()
    '''    
    def make_test(self):
        N_test, D = self.XY_test.shape
        new_test = np.empty((200, 2*D-1))
        count = 0
        pos = 0
        neg = 0
        for i in xrange(N_test):
            for j in xrange(i+1,N_test):
                new_example = np.empty((1,2*D-1))
                new_example[0,:(D-1)] = self.XY_test[i,:-1]
                new_example[0,(D-1):-1] = self.XY_test[j,:-1]
                if self.XY_test[i,-1] == self.XY_test[j,-1]:
                    if pos >= 100:
                        continue
                    pos += 1
                    count += 1
                    new_example[0,-1] = 1
                else:
                    if neg >= 100:
                       continue
                    neg += 1
                    count += 1
                    new_example[0,-1] = 0
                if count >= 200:
                    continue
                new_test[count] = new_example
        self.XY_test = new_test
        print self.XY_test.shape
        print 'count: %d (pos: %d, neg: %d)' %(count, pos, neg)
        sys.stdout.flush()
    '''
    def make_test(self, n):
        N_test, D = self.XY_test.shape
        new_test = np.empty((n, 2*D-1))
        count = 0
        pos = 0
        neg = 0
        while count < n:
            new_example = np.empty((1,2*D-1))
            i = int(np.random.rand() * N_test)
            j = int(np.random.rand() * N_test)
            new_example[0,:(D-1)] = self.XY_test[i,:-1]
            new_example[0,(D-1):-1] = self.XY_test[j,:-1]
            if self.XY_test[i,-1] == self.XY_test[j,-1] and pos < n/2:
                pos += 1                
                new_example[0,-1] = 1
                new_test[count] = new_example
                count += 1
            elif self.XY_test[i,-1] != self.XY_test[j,-1] and neg < n/2:
                neg += 1
                new_example[0,-1] = 0
                new_test[count] = new_example
                count += 1
        self.XY_test = new_test
        print self.XY_test.shape
        print 'count: %d (pos: %d, neg: %d)' %(count, pos, neg)
        sys.stdout.flush()


    def test(self):
        self.make_test(200)
        N, D = self.XY_test.shape
        Y = self.XY_test[:,-1]
        pred = self.classify()
        TP = FP = FN = TN = 0
        res = np.empty((N,))
        for i in xrange(N):
            l1 = pred[i*2]
            l2 = pred[i*2+1]
            if l1 == l2 and Y[i] == 1:
                TP += 1
            elif l1 == l2 and Y[i] != 1:
                FP += 1
            elif l1 != l2 and Y[i] == 1:
                FN += 1
            elif l1 != l2 and Y[i] != 1:
                TN += 1
            res[i] = np.bitwise_xor(int(np.sign(abs(l1-l2))),int(Y[i]))
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
            #print [self.entity_pair_labels[int(j)] for j in top_indices]
            pred[i] = stats.mode([self.entity_pair_labels[int(j)] for j in top_indices])[0][0]
        return pred
            
    def build_dist_matrix(self):        
        fname = 'distance_matrix_%s.npy' %self.rep
        if os.path.isfile(fname):
            print 'loading %s . . .' %fname
            self.distances = np.load(fname)
        else:
            print 'building distance matrix . . .'
            d, _ = self.entity_pairs.shape
            N_test, D = self.XY_test.shape
            self.distances = np.empty((2*N_test,d))
            print N_test
            for i in xrange(N_test):
                print 'row',i 
                x1 = self.XY_test[i,:(D-1)/2]
                x2 = self.XY_test[i,(D-1)/2:-1]
                self.distances[2*i] = np.array([self._distance(x1,self.entity_pairs[j]) for j in xrange(d)])
                self.distances[2*i-1] = np.array([self._distance(x2,self.entity_pairs[j]) for j in xrange(d)])            
            np.save(fname, self.distances)
        print 'distances: ',self.distances.shape
        sys.stdout.flush()
            
    def build_top_indices(self):
        if os.path.isfile('top_indices_%s.npy' %self.rep):
            print 'loading top_indices . . .'
            self.top_indices = np.load('top_indices_%s.npy' %self.rep)
            sys.stdout.flush()
            return
        print 'creating top_indices . . .'
        self.build_dist_matrix()
        N_test, N_train = self.distances.shape
        self.top_indices = np.empty((N_test, N_train))
        for j in xrange(N_test):
            row = self.distances[j]
            tagged_row = [(row[i], i) for i in xrange(N_train)]
            tagged_row.sort(key=lambda d: d[0]) # sort by distance
            self.top_indices[j] = np.array([d[1] for d in tagged_row])
        #print 'top_indices',self.top_indices.shape
        np.save('top_indices_%s.npy' %self.rep, self.top_indices)
        sys.stdout.flush()
        
    def _vec2str(self, arr):
        return str(list(arr))
    
    def _str2vec(self, str):
        return np.array(eval(str))    

    def _distance(self, u, v):
        return sps.distance.euclidean(u,v)
    

# Disjoint set related methods
class NodeSet:
    def __init__(self, arr):
        self.parent = self
        self.rank = 0
        self.arr = arr
        
def find(x):
    if x.parent != x:
        x.parent = find(x.parent)
    return x.parent

def union(x, y):
    x_root = find(x)
    y_root = find(y)
    if x_root == y_root:
        return # already the same set
    if x_root.rank < y_root.rank:
        x_root.parent = y_root
    elif x_root.rank > y_root.rank:
        y_root.parent = x_root
    else:
        y_root.parent = x_root
        x_root.rank += 1

if __name__=='__main__':
    for k in [5,10,15]:
        for rep in ['','subtract','autoencode']:
            print 'k =',k
            print 'rep:',rep
            knn = KNN(k,rep)
            knn.train()
            knn.test()
