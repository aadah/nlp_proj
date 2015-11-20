import h5py
import numpy as np
from numpy import dtype, fromstring, float32
from gensim import utils


# move this file to ~/Documents/nlp_resource/word2vec/
# go to that directory and type:
#
#    python convert.py > entities
#
# it creates the hdf5 file for you in the same directory. if also creates
# the entities file, which is required for the model. make sure
# the freebase-vectors-skipgram1000-en.bin.gz is in the directory as well!
# you can get it from https://docs.google.com/uc?id=0B7XkCwpI5KDYeFdmcVltWkhtbmM&export=download
# you only need to run this ONCE. it will take nearly an hour.
#
# once created, you can use vector.py to access the vectors in a
# memory effiecient way


def main(filename):
    fn = filename.split('.')
    fn[-1] = 'hdf5'
    hdf5_filename = '.'.join(fn)

    bin_f = utils.smart_open(filename)
    hdf5_f = h5py.File(hdf5_filename, 'w')

    header = utils.to_unicode(bin_f.readline(), encoding='utf8')
    vocab_size, vector_size = map(int, header.split())
    binary_len = dtype(float32).itemsize * vector_size
    
    for line_no in xrange(vocab_size):
        word = []

        while True:
            ch = bin_f.read(1)
            if ch == b' ':
                break
            if ch != b'\n':
                word.append(ch)

        word = utils.to_unicode(b''.join(word), encoding='utf8', errors='strict')
        vector = fromstring(bin_f.read(binary_len), dtype=float32)

        if word[:3] == '/en':
            w = word[4:]
            hdf5_f.create_dataset(w, data=vector)
            print w

    bin_f.close()
    hdf5_f.close()


main('freebase-vectors-skipgram1000-en.bin.gz')
