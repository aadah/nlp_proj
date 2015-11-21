#!/usr/bin/python
import collections
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import vector


class RelationCluster():

    def __init__(entity_set, eps=0.5, min_samples=5, metric='euclidean'):
        self.vm = VectorModel()
        self.cluster = DBSCAN(esp=esp, min_samples=min_samples, metric=metric)
        self.entity_set = entity_set        
        self.X, self.relation_index = self.get_relation_vecs(entity_set)
        self.labels = self.cluster.fit(self.X)
        
    def get_relation_vec(self, w1, w2):
        return self.vm.vector(w1) - self.vm.vector(w2)

    def get_relation_vecs(self):
        relation_index = {}
        relation_list = []
        index = 0
        for entity1 in entity_set:
            for entity2 in entity_set:
                if entity1 == entity2:
                    continue
                relation_index[(entity1, entity2)] = index
                relation_list.add(get_relation_vec(entity1, entity2))
                index += 1
        X = np.array(relation_list)
        return X, relation_index

    def get_num_clusters(self):
        return len(set(self.labels)) - (1 if -1 in self.labels else 0)
    
    def get_unique_labels(self):
        return set(self.labels)

    def get_label(self, pair):
        return self.labels[self.relation_index[pair]]

    def is_core(self, pair):
        return self.relation_index[pair] in self.cluster.core_sample_indices_

    def get_cluster_sizes(self):
        counts = {}
        for label in self.get_unique_labels():
            counts[label] = self.labels.count(label)
        return counts

    def get_vector_clusters(self):
        clusters = collections.defaultdict([])
        for i in self.cluster.core_sample_indices_:
            clusters[self.labels[i]].append(self.X[i])
        return clusters

    def get_vector_cluster_means(self):
        pass #TODO

    def get_vector_cluster_variances(self):
        pass #TODO
