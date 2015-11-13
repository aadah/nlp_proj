import os

HOME = os.path.expanduser("~")
DOCUMENTS = "%s/Documents" % HOME
RESOUCRES = "%s/nlp_resources" % DOCUMENTS

# word2vec
WORD2VEC_DIR = "%s/word2vec" % RESOUCRES
FREEBASE_FILE = "%s/knowledge-vectors-skipgram1000.bin" % WORD2VEC_DIR
