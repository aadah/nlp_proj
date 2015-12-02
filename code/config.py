import os

HOME = os.path.expanduser("~")
DOCUMENTS = "%s/Documents" % HOME
RESOURCES = "%s/nlp_resources" % DOCUMENTS

# word2vec
WORD2VEC_DIR = "%s/word2vec" % RESOURCES
ENTITIES_FILE = "%s/entities" % WORD2VEC_DIR
MAPPINGS_FILE = "%s/mappings" % WORD2VEC_DIR
FREEBASE_FILE = "%s/freebase-vectors-skipgram1000-en.bin.hdf5" % WORD2VEC_DIR
MID_FREEBASE_FILE = "%s/freebase-vectors-skipgram1000.bin.hdf5" % WORD2VEC_DIR

# reuters
REUTERS_DIR = "%s/reuters21578" % RESOURCES
PAIR_FILE = "%s/entity_pairs.txt" % REUTERS_DIR
REUTERS_PAIRS = "%s/reuters_entity_pairs_stanford_1000.txt" % REUTERS_DIR

# bbc
BBC_DIR = "%s/bbc" % RESOURCES
BBC_ARTICLES = "bbc_links.txt"
BBC_PAIRS = "%s/entity_pairs.txt" % BBC_DIR

# stanford
STANFORD_NER_DIR = "%s/stanford-ner-2014-08-27" % RESOURCES
STANFORD_NER_JAR = "%s/stanford-ner.jar" % STANFORD_NER_DIR
STANFORD_NER_MODELS_DIR = "%s/classifiers" % STANFORD_NER_DIR
STANFORD_3CLASS = '%s/english.all.3class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
STANFORD_4CLASS = '%s/english.conll.4class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
STANFORD_7CLASS = '%s/english.muc.7class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR

# wikidata
WIKIDATA_ENDPOINT = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
WIKIDATA_QUERY_TEMPLATE = 'https://www.wikidata.org/w/index.php?search=%s&search=%s&title=Special%%3ASearch&go=Go'

# parameters
NER_MODE = 'stanford'
ARTICLE_SOURCE = 'bbc' # can be 'reuters'
######## Change this to produce new pairs if you update bbc_links.txt
READ_FROM_PAIRS = False
########
STANFORD_MODEL_NUM = 3
PAIR_OUTPUT_FILE = BBC_PAIRS
PAIR_INPUT_FILE = BBC_PAIRS

CLUSTER_EPS = 0.7
CLUSTER_MIN_SAMPLES = 3
CLUSTER_METRIC = 'euclidean'
CLUSTER_ALGO = 'ball_tree'
