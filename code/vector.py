import re
import h5py
import scipy.spatial as sps
import numpy as np

import config


class VectorModel:
    """Creates entity vector model. Interface is very similary to a dictionary:

    # create model
    model = VectorModel()
    
    # the following return the same vector
    model['Barack Obama']
    model['barack   obama']
    model['BARACK OBAMA']
    model['bArack  ObaMa']
    model['barack_obama']

    # you can check for membership:
    'vladimir putin' in model

    # in keyword only for boolean expression, not control flow:
    if 'Marvin Minsky' in model: do_something() # OK
    for entity in model: do_something(entity)   # NOT OK

    # you can check similarity between things on a scale from 0 to 1
    model.similarity('world war i', 'world war ii')
    """

    def __init__(self):
        self.f = h5py.File(config.FREEBASE_FILE)
        
        with open(config.ENTITIES_FILE) as f:
            self.entities = [ent.strip() for ent in f.readlines()]


    def vector(self, string):
        if string == '':
            return None # bug where it hangs on empty string
        elif string not in self:
            return None

        return self._vector(self._format_string(string))


    def _vector(self, string):
        return np.array(self.f[string])


    def similarity(self, s1, s2):
        u = self[s1]
        v = self[s2]

        if u is None or v is None:
            return 0.0

        return self._similarity(u, v)


    def _similarity(self, u, v):
        return 1 - sps.distance.cosine(u, v)


    def most_similar(self, string, k=1):
        if string not in self:
            return []

        u = self[string]

        return self._most_similar(u, self._format_string(string), k=k)


    def _most_similar(self, u, string, k=1):
        best = [(None, float('-inf')) for _ in xrange(k)]

        for ent in self.entities:
            if ent == string: continue

            v = self._vector(ent)
            sim = self._similarity(u, v)
            _, lowest_sim = best[0]

            if sim > lowest_sim:
                best[0] = (ent, sim)
                self._move_up(best)

        best.reverse()

        return best


    def _move_up(self, l):
        k = len(l)

        for i in xrange(k-1):
            elem = l[i]
            next_elem = l[i+1]
            
            if elem[1] > next_elem[1]:
                l[i] = next_elem
                l[i+1] = elem
                continue
            else:
                break


    def _format_string(self, string):
        string = string.lower()
        string = self._nonalphanumeric_to_underscore(string)
        
        return string


    def _nonalphanumeric_to_underscore(self, string):
        string = re.sub(r'[^a-z0-9-]+', '_', string)
        
        return string


    def close(self):
        self.f.close()


    def __getitem__(self, string):
        return self.vector(string)


    def __contains__(self, string):
        return self._format_string(string) in self.f
