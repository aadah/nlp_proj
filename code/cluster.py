#!/usr/bin/python
import collections
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from vector import VectorModel


class RelationCluster():

    def __init__(self, entity_set, eps=1.5, min_samples=3, metric='euclidean'):
        self.vm = VectorModel()
        self.cluster = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        self.entity_set = entity_set
        self.X, self.relation_index = self.get_relation_vecs()
        self.cluster.fit(self.X)
        self.labels = self.cluster.labels_
        
    def get_relation_vec(self, w1, w2):
        # Gets the difference between the word vector for w1 and the word vector for w2
        return self.vm.vector(w1) - self.vm.vector(w2)

    def get_relation_vecs(self):
        # Gets the relation vectors between every pair in the set
        relation_index = {}
        relation_list = []
        index = 0
        for entity1 in self.entity_set:
            for entity2 in self.entity_set:
                if entity1 == entity2:
                    continue
                relation_index[(entity1, entity2)] = index
                relation_index[index] = (entity1, entity2)
                relation_list.append(self.get_relation_vec(entity1, entity2))
                index += 1
        X = np.array(relation_list)
        return X, relation_index

    def get_num_clusters(self):
        # Returns the number of clusters found
        # The set of relation vectors that do not belong in any cluster 
        # does not count as a cluster
        return len(set(self.labels)) - (1 if -1 in self.labels else 0)
    
    def get_unique_labels(self):
        # Returns the set of unique labels, or the index of each cluster
        return set(self.labels)

    def get_label(self, pair):
        # Returns the label of the cluster that a pair's relation vector belongs in
        return self.labels[self.relation_index[pair]]

    def is_core(self, pair):
        # Returns whether a pair's relation vector is part of a cluster
        return self.relation_index[pair] in self.cluster.core_sample_indices_

    def get_cluster_sizes(self):
        # Returns a dictionary mapping cluster labels to the cluster sizes
        counts = {}
        label_list = list(self.labels)
        for label in self.get_unique_labels():
            counts[label] = label_list.count(label)
        return counts

    def get_vector_clusters(self):
        # Returns a tuple of two dictionaries
        # The first dictionary maps labels to lists of relation vectors
        # The second dictionary maps labels to the indices of the relation vectors that are
        # in the cluster of that label
        clusters = collections.defaultdict(list)
        index_clusters = collections.defaultdict(list)
        core_indices = self.cluster.core_sample_indices_
        for i in core_indices:
            clusters[self.labels[i]].append(self.X[i])
            index_clusters[self.labels[i]].append(i)
        return clusters, index_clusters

    def get_vector_cluster_means(self):
        clusters, index_clusters = self.get_vector_clusters()
        means = {}
        for label in self.get_unique_labels():
            cluster = np.array(clusters[label])
            print cluster[label].shape
            mu = cluster.mean(axis=0)
            means[label] = mu
        return means

    def get_vector_cluster_variances(self):
        # finds the variance from the mean 
        pass

    def print_clusters(self):
        if len(self.cluster.core_sample_indices_) == 0:
            print "NO CLUSTERS :("
            return
        clusters, index_clusters = self.get_vector_clusters()
        cluster_sizes = self.get_cluster_sizes()
        cluster_vector_means = self.get_vector_cluster_means()
        for label in self.get_unique_labels():
            if label == -1:
                continue
            print "Cluster: %d" % label
            print "cluster size: %d" % cluster_sizes[label]
            mean = cluster_vector_means[label]
            '''
            for index in index_clusters[label]:
                print '\t' + str(self.relation_index[index])
                print ('\t euclidean diff from mean: ' + 
                       str(metrics.pairwise.pairwise_distances(clusters[index],mean)))
                       '''

if __name__=="__main__":
    entityWords = ['obama','america','washington_d_c',
                   'stephen_harper','canada','ottawa',
                   'david_cameron','england','london',
                   'vladimir_putin','russia','moscow',
                   'francois_hollande','france','paris',
                   'angela_merkel','germany','berlin',
                   'enrique_pena_nieto','mexico','mexico_city',
                   'deng_xiaoping','china','beijing',
                   'spain','madrid',
                   'belgium','brussels',
                   'taiwan','taipei',
                   'korea','seoul',
                   'abe','japan','tokyo'
                   ]
    rc = RelationCluster(entityWords)
    print rc.get_num_clusters()
    print rc.get_unique_labels()
    rc.print_clusters()
