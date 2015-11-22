import os

HOME = os.path.expanduser("~")
DOCUMENTS = "%s/Documents" % HOME
RESOURCES = "%s/nlp_resources" % DOCUMENTS

# word2vec
WORD2VEC_DIR = "%s/word2vec" % RESOURCES
ENTITIES_FILE = "%s/entities" % WORD2VEC_DIR
FREEBASE_FILE = "%s/freebase-vectors-skipgram1000-en.bin.hdf5" % WORD2VEC_DIR

# reuters
REUTERS_DIR = "%s/reuters21578" % RESOURCES
PAIR_FILE = "%s/entity_pairs.txt" % REUTERS_DIR
