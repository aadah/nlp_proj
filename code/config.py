import os

HOME = os.path.expanduser("~")
DOCUMENTS = "%s/Documents" % HOME
RESOUCRES = "%s/nlp_resources" % DOCUMENTS

# word2vec
WORD2VEC_DIR = "%s/word2vec" % RESOUCRES
ENTITIES_FILE = "%s/entities" % WORD2VEC_DIR
FREEBASE_FILE = "%s/freebase-vectors-skipgram1000-en.bin.hdf5" % WORD2VEC_DIR
