import h5py
from gensim import utils
import numpy as np
from numpy import dtype, fromstring, float32

# move this file to ~/Documents/nlp_resource/word2vec/
# go to that directory and type:
#
#    python convert.py
#
# it creates the hdf5 file for you in the same directory. make sure
# the freebase-vectors-skipgram1000-en.bin.gz is in the directory as well!
# you can get it from https://docs.google.com/uc?id=0B7XkCwpI5KDYeFdmcVltWkhtbmM&export=download
# you only need to run this ONCE. it will take nearly an hour.
#
# once created, you can use vector.py to access the vectors in a
# memory effiecient way


def main_(filename):
    fn = filename.split('.')
    fn[-1] = 'hdf5'
    hdf5_filename = '.'.join(fn)

    text_f = open(filename, 'r')
    hdf5_f = h5py.File(hdf5_filename, 'w')

    s = text_f.readline().strip().split()

    while len(s) != 0:
        word = s[0]
        vector = np.array([float(num) for num in s[1:]])

        hdf5_f.create_dataset(word, data=vector)

        s = text_f.readline().strip().split()

    text_f.close()
    hdf5_f.close()


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
        hdf5_f.create_dataset(word, data=vector)

    bin_f.close()
    hdf5_f.close()


#main_('vectors_wiki.normalized.txt')
main('freebase-vectors-skipgram1000-en.bin.gz')
