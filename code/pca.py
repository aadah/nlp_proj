import config
import collections
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from heapq import nlargest

class Compressor:
    def __init__(self, rep=''):
        print 'Handling %s' %rep
        self.rep = rep
        if rep == '':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_NPY)
            self.f_train = '%s/pca_data_train.npy' % config.RESOURCES
            self.f_test = '%s/pca_data_test.npy' % config.RESOURCES
        elif rep == 'subtract':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_SUBTRACT_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_SUBTRACT_NPY)
            self.f_train = '%s/pca_data_train_subtract.npy' % config.RESOURCES
            self.f_test = '%s/pca_data_test_subtract.npy' % config.RESOURCES
        elif rep == 'autoencode':
            self.XY_train = np.load(config.NEW_DATA_TRAIN_AUTOENCODE_NPY)
            self.XY_test = np.load(config.NEW_DATA_TEST_AUTOENCODE_NPY)
            self.f_train = '%s/pca_data_train_autoencode.npy' % config.RESOURCES
            self.f_test = '%s/pca_data_test_autoencode.npy' % config.RESOURCES
        else:
            print "invalid representation input! Please choose '', 'subtract', or 'autoencode'"
    
    def compress(self, d):
        X_train, Y_train = self.separate_XY(self.XY_train)
        X_test, Y_test = self.separate_XY(self.XY_test)
        
        pca = PCA(n_components=d)
        X_train_set = self.build_relation_set(X_train)
        pca.fit(X_train_set)
        
        # transform X_train
        print 'Compressing training data . . .'
        N, D = X_train.shape
        X1_train = X_train[:,:D/2]
        X2_train = X_train[:,D/2:]
        new_XY_train = np.empty((N, 2*d + 2))
        new_XY_train[:,:d] = pca.transform(X1_train)
        new_XY_train[:,d:-2] = pca.transform(X2_train)
        new_XY_train[:,-2:] = Y_train
        print 'dimensions:', new_XY_train.shape
        np.save(self.f_train, new_XY_train)

        # transform X_test
        print 'Compressing testing data . . .'
        N, D = X_test.shape
        X1_test = X_test[:,:D/2]
        X2_test = X_test[:,D/2:]
        new_XY_test = np.empty((N, 2*d + 2))
        new_XY_test[:,:d] = pca.transform(X1_test)
        new_XY_test[:,d:-2] = pca.transform(X2_test)
        new_XY_test[:,-2:] = Y_test
        print 'dimensions:', new_XY_test.shape
        np.save(self.f_test, new_XY_test)
        
    def separate_XY(self, XY):
        X = XY[:,:-2]
        Y = XY[:,-2:]
        return (X, Y)

    def build_relation_set(self, X):
        N, D = X.shape
        relation_set = set()
        for i in xrange(N):
            x1 = X[i,:D/2]
            x2 = X[i,D/2:]
            x1_str = vec2str(x1)
            x2_str = vec2str(x2)
            if x1_str not in relation_set:
                relation_set.add(x1_str)
            if x2_str not in relation_set:
                relation_set.add(x2_str)
        N_set = len(relation_set)
        print '# unique relations:',N_set
        unique_relations = np.empty((N_set,D/2))
        i = 0
        for relation in relation_set:
            x = str2vec(relation)
            unique_relations[i] = x
            i += 1
        return unique_relations


class Visualizer:
    
    def __init__(self, XY=None, rep=''):
        self.rep = rep
        if rep == '':
            self.rep = 'concatenate'
            self.XY = np.load(config.NEW_DATA_TRAIN_NPY)
        elif rep == 'subtract':
            self.XY = np.load(config.NEW_DATA_TRAIN_SUBTRACT_NPY)
        elif rep == 'autoencode':
            self.XY = np.load(config.NEW_DATA_TRAIN_AUTOENCODE_NPY)
        elif rep == 'pca':
            self.rep = 'pca_concatenate'
            self.XY = np.load(config.PCA_DATA_TRAIN_NPY)
        elif rep == 'pca_subtract':
            self.XY = np.load(config.PCA_DATA_TRAIN_SUBTRACT_NPY)
        elif rep == 'pca_autoencode':
            self.XY = np.load(config.PCA_DATA_TRAIN_AUTOENCODE_NPY)
        print self.rep
        self.X = self.XY[:,:-1]
        self.Y = self.XY[:,-1]
        
    def sanity_check(self):
        relation_set = self.make_set()
        relation_count = collections.defaultdict(int)
        for relation in relation_set:
            relation_count[relation[-1]] += 1
        for relation_id in relation_count:
            if relation_id != 0:
                print '%d: %d' %(relation_id, relation_count[relation_id])
        return relation_count

    def make_set(self):
        N, D = self.XY.shape
        relation_set = set()
        for i in xrange(N):
            x1 = self.XY[i,range(0,(D-1)/2)+range(D-2,D-1)]
            x2 = self.XY[i,range((D-1)/2,D-2)+range(D-1,D)]
            x1_str = vec2str(x1)
            x2_str = vec2str(x2)
            if x1_str not in relation_set:
                relation_set.add(x1_str)
            if x2_str not in relation_set:
                relation_set.add(x2_str)
        N_set = len(relation_set)
        print 'set length:',N_set
        unique_relations = np.empty((N_set,(D-1)/2+1))
        i = 0
        for relation in relation_set:
            x = str2vec(relation)
            unique_relations[i] = x
            i += 1
        return unique_relations

    def PCA_transform(self, n_components, overwrite=False):
        fname = 'PCA_results/pca_%s_%d-comp.npy' %(self.rep, n_components)
        if os.path.isfile(fname) and not overwrite:
            print 'Loading %s . . .' % fname
            self.XY_pca = np.load(fname)
            self.X_pca = self.XY_pca[:,:-1]
        else:
            print 'Running PCA . . .'
            pca = PCA(n_components=n_components)
            #self.X_pca = pca.fit_transform(self.X)
            unique_relations = self.make_set()
            self.X_pca = pca.fit_transform(unique_relations[:,:-1])
            self.XY_pca = np.empty((unique_relations.shape[0],self.X_pca.shape[1]+1))
            self.XY_pca[:,:-1] = self.X_pca
            self.XY_pca[:,-1] = unique_relations[:,-1]
            np.save(fname, self.XY_pca)

    def plot_X(self, n_components, r1, r2, r3):
        self.PCA_transform(n_components)
        
        fig = plt.figure()
        N, _ = self.XY_pca.shape
        y = self.XY_pca[:,-1]
        y = y.reshape((y.shape[0],)) # in order for boolean mask to work
        
        props = config.WIKIDATA_PROPERTIES_DICT
        
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            for c, i, target_name in zip("rgb", [r1,r2,r3], ['P%d'%r1,'P%d'%r2, 'P%d'%r3]):
                ax.scatter(self.X_pca[y == i, 0], self.X_pca[y == i, 1], zs=self.X_pca[y == i, 2], c=c, label=target_name)
            ax.set_title('PCA w/ %s - %s(%d) vs. %s(%d) vs. %s(%d)' % (self.rep, props[r1],r1, props[r2],r2, props[r3],r3))
        else:
            for c, i, target_name in zip("rgb", [r1,r2,r3], ['P%d'%r1,'P%d'%r2, 'P%d'%r3]):
                plt.scatter(self.X_pca[y==i, 0], self.X_pca[y==i, 1], c=c, label=target_name)
            plt.title('PCA w/ %s - %s(%d) vs. %s(%d) vs. %s(%d)' % (self.rep, props[r1],r1, props[r2],r2, props[r3],r3))
        plt.legend()
        plt.show()
        
def vec2str(arr):
    return str(list(arr))
    
def str2vec(str):
    return np.array(eval(str))

def compress_all():    
    for rep in ['', 'subtract', 'autoencode']:
        comp = Compressor(rep=rep)
        comp.compress(100)

if __name__=="__main__":
    #compress_all()

    for rep in ['', 'subtract', 'autoencode','pca','pca_subtract','pca_autoencode']:
        vis = Visualizer(rep=rep)
        #vis.plot_X(3, 26, 451, 36) # spouse vs. partner vs. capital
        #vis.plot_X(2, 26, 451, 36) # spouse vs. partner vs. capital
        #vis.plot_X(3, 22, 25, 38) # father vs. mother vs. currency
        vis.plot_X(2, 22, 25, 38) # father vs. mother vs. currency
        #vis.plot_X(3, 7, 9, 6) # brother vs. sister vs. head of gov't
        #vis.plot_X(2, 7, 9, 6) # brother vs. sister vs. head of gov't
        #vis.plot_X(3, 200, 201, 802) # lake inflow vs. lake outflow vs. student
    #vis.plot_X(2, 200, 201, 802) # lake inflow vs. lake outflow vs. student
    #vis.plot_X('earn', n_components=3)
    #vis.sanity_check()
    #vis.plot_X(3, 0, 1)

    

