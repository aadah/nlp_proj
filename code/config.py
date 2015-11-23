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

# stanford
STANFORD_NER_DIR = "%s/stanford-ner-2014-08-27" % RESOURCES
STANFORD_NER_JAR = "%s/stanford-ner.jar" % STANFORD_NER_DIR
STANFORD_NER_MODELS_DIR = "%s/classifiers" % STANFORD_NER_DIR
STANFORD_3CLASS = '%s/english.all.3class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
STANFORD_4CLASS = '%s/english.conll.4class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
STANFORD_7CLASS = '%s/english.muc.7class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
