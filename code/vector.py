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


    def vector(self, string):
        if string == '': return None # bug where it hangs on empty string

        string = self._format_string(string)

        if string in self.f:
            return np.array(self.f[string])

        return None # there is no vector


    def similarity(self, s1, s2):
        v1 = self[s1]
        v2 = self[s2]

        if v1 != None and v2 != None:
            return 1 - sps.distance.cosine(v1, v2)

        return 0.0


    def _format_string(self, string):
        string = string.lower()
        string = self._nonalphanumeric_to_underscore(string)
        
        return '/en/%s' % string


    def _nonalphanumeric_to_underscore(self, string):
        string = re.sub(r'\W+', '_', string)
        
        return string


    def close(self):
        self.f.close()


    def __getitem__(self, string):
        return self.vector(string)


    def __contains__(self, string):
        return self._format_string(string) in self.f
