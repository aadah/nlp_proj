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
BBC_DEP_PARSER_PAIRS = "%s/dep_parser_entity_pairs2.txt" % BBC_DIR

# stanford
STANFORD_NER_DIR = "%s/stanford-ner-2014-08-27" % RESOURCES
STANFORD_NER_JAR = "%s/stanford-ner.jar" % STANFORD_NER_DIR
STANFORD_NER_MODELS_DIR = "%s/classifiers" % STANFORD_NER_DIR
STANFORD_3CLASS = '%s/english.all.3class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
STANFORD_4CLASS = '%s/english.conll.4class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR
STANFORD_7CLASS = '%s/english.muc.7class.distsim.crf.ser.gz' % STANFORD_NER_MODELS_DIR

STANFORD_PARSER_DIR = "%s/stanford-parser-full-2015-04-20" % RESOURCES
STANFORD_PARSER_JAR = "%s/stanford-parser.jar" % STANFORD_PARSER_DIR
STANFORD_PARSER_MODEL = "%s/stanford-parser-3.5.2-models.jar" % STANFORD_PARSER_DIR

# data
RAW_DATA_TXT = '%s/raw_data.txt' % RESOURCES
NEW_DATA_TXT = '%s/new_data.txt' % RESOURCES

NEW_DATA_TRAIN_NPY = '%s/new_data_train.npy' % RESOURCES
NEW_DATA_TEST_NPY = '%s/new_data_test.npy' % RESOURCES
NEW_DATA_TRAIN_SUBTRACT_NPY = '%s/new_data_train_subtract.npy' % RESOURCES
NEW_DATA_TEST_SUBTRACT_NPY = '%s/new_data_test_subtract.npy' % RESOURCES
NEW_DATA_TRAIN_AUTOENCODE_NPY = '%s/new_data_train_autoencode.npy' % RESOURCES
NEW_DATA_TEST_AUTOENCODE_NPY = '%s/new_data_test_autoencode.npy' % RESOURCES

PCA_DATA_TRAIN_NPY = '%s/pca_data_train.npy' % RESOURCES
PCA_DATA_TEST_NPY = '%s/pca_data_test.npy' % RESOURCES
PCA_DATA_TRAIN_SUBTRACT_NPY = '%s/pca_data_train_subtract.npy' % RESOURCES
PCA_DATA_TEST_SUBTRACT_NPY = '%s/pca_data_test_subtract.npy' % RESOURCES
PCA_DATA_TRAIN_AUTOENCODE_NPY = '%s/pca_data_train_autoencode.npy' % RESOURCES
PCA_DATA_TEST_AUTOENCODE_NPY = '%s/pca_data_test_autoencode.npy' % RESOURCES

PCA_OUT = '%s/pca_out.txt' % RESOURCES

# params
NEW_DATA_PARAMS = '%s/new_data.hd5f' % RESOURCES
NEW_DATA_SUBTRACT_PARAMS = '%s/new_data_subtract.hd5f' % RESOURCES
NEW_DATA_AUTOENCODE_PARAMS = '%s/new_data_autoencode.hd5f' % RESOURCES

#####

RELATIONS_DICT = '%s/relations_dictionary.txt' % RESOURCES
SINGLE_INSTANCE_TRAIN = '%s/single_instance_train.npy' % RESOURCES
SINGLE_INSTANCE_TEST = '%s/single_instance_test.npy' % RESOURCES
SUBTRACT_SINGLE_INSTANCE_TRAIN = '%s/subtract_single_instance_train.npy' % RESOURCES
SUBTRACT_SINGLE_INSTANCE_TEST = '%s/subtract_single_instance_test.npy' % RESOURCES
AUTOENCODE_SINGLE_INSTANCE_TRAIN = '%s/autoencode_single_instance_train.npy' % RESOURCES
AUTOENCODE_SINGLE_INSTANCE_TEST = '%s/autoencode_single_instance_test.npy' % RESOURCES

TWO_INSTANCE_FILE = '%s/two_instance.txt' % RESOURCES
TWO_INSTANCE_DATA = '%s/two_instance.npy' % RESOURCES
AUTOENCODER_FILE = '%s/autoencoder.txt' % RESOURCES
AUTOENCODER_DATA = '%s/autoencoder.npy' % RESOURCES

# neural net
TWO_INSTANCE_PARAMS = '%s/two_instance_params.hdf5' % RESOURCES
TWO_INSTANCE_PARAMS_CONNECT = '%s/two_instance_params_connect.hdf5' % RESOURCES
SUBTRACT_TWO_INSTANCE_PARAMS_CONNECT = '%s/two_instance_params_connect.hdf5' % RESOURCES
AUTOENCODE_TWO_INSTANCE_PARAMS_CONNECT = '%s/two_instance_params_connect.hdf5' % RESOURCES
AUTOENCODER_PARAMS = '%s/autoencoder_params.hdf5' % RESOURCES
SIMPLE_AUTOENCODER_PARAMS = '%s/simple_autoencoder_params.hdf5' % RESOURCES

# parameters
NER_MODE = 'stanford'
PARSER = 'stanford_dep_parser'
ARTICLE_SOURCE = 'bbc' # can be 'reuters'
######## Change this to produce new pairs if you update bbc_links.txt
#READ_FROM_PAIRS = True # Will not generate new pairs but rather simply reads from PAIR_INPUT_FILE
READ_FROM_PAIRS = False # Will generate new pairs and write to PAIR_OUTPUT_FILE
########
STANFORD_MODEL_NUM = 3
PAIR_OUTPUT_FILE = BBC_DEP_PARSER_PAIRS
PAIR_INPUT_FILE = BBC_DEP_PARSER_PAIRS

CLUSTER_EPS = 0.7
CLUSTER_MIN_SAMPLES = 3
CLUSTER_METRIC = 'euclidean'
CLUSTER_ALGO = 'ball_tree'

# SVM results directory
SVM_RESULTS_DIR = '%s/svm_results/' % RESOURCES

# Split data files
DATA_TRAIN = '%s/split_train_95.npy' % RESOURCES
DATA_TEST  = '%s/split_test_95.npy' % RESOURCES
SUBTRACT_DATA_TRAIN = '%s/subtract_split_train_95.npy' % RESOURCES
SUBTRACT_DATA_TEST  = '%s/subtract_split_test_95.npy' % RESOURCES
AUTOENCODE_DATA_TRAIN = '%s/autoencode_split_train_95.npy' % RESOURCES
AUTOENCODE_DATA_TEST  = '%s/autoencode_split_test_95.npy' % RESOURCES

# wikidata
WIKIDATA_ENDPOINT = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
WIKIDATA_QUERY_TEMPLATE = 'https://www.wikidata.org/w/index.php?search=%s&search=%s&title=Special%%3ASearch&go=Go'
WIKIDATA_PROPERTIES = [
    # people properties
    'P19', # place of birth
    'P20', # place of death
    'P509', # cause of death
    'P157', # killed by
    'P119', # place of burial
    'P66', # ancestral home
    'P463', # member of
    'P172', # ethnic group
    'P103', # native language
    'P27', # country of citizenship
    'P69', # educated at
    'P106', # occupation
    'P101', # field of work
    'P800', # notable work
    'P108', # employer
    'P39', # position held
    'P102', # member of political party
    'P263', # official residence
    'P140', # religion
    'P91', # sexual orientation
    'P184', # doctoral advisor
    'P185', # doctoral student
    'P1066', # student of
    'P802', # student
    'P53', # noble family
    'P97', # noble title
    'P512', # academic degree
    'P1412', # affiliation
    'P1429', # pet
    'P22', # father
    'P25', # mother
    'P7', # brother
    'P9', # sister
    'P26', # spouse
    'P451', # partner
    'P40', # child
    'P43', # stepfather
    'P44', # stepmother
    'P1038', # relative
    'P1290', # godparent
    'P54', # member of sports team
    'P241', # military branch
    'P410', # military rank
    'P598', # commander of
    'P607', # conflict

    # organization properties
    'P159', # headquarters location
    'P452', # industry
    'P807', # separated from
    'P1056', # product
    'P457', # foundational text
    'P112', # founder
    'P740', # location of formation
    'P1387', # political alignment
    'P1408', # licensed to broadcast to
    'P1313', # office held by head of government
    'P113', # airline hub
    'P114', # airline alliance
    'P115', # home venue
    'P118', # league
    'P286', # head coach
    'P505', # general manager
    'P634', # captain
    'P822', # mascot
    'P169', # chief executive officer
    'P488', # chairperson
    'P199', # business division
    'P355', # subsidiaries
    'P1268', # represents organisation
    'P1308', # officeholder
    'P1320', # stock exchange
    'P1037', # manager/director

    # place properties
    'P163', # flag
    'P417', # patron saint
    'P30', # continent
    'P17', # country
    'P131', # located in the administrative territorial entity
    'P421', # located in time zone
    'P706', # located on terrain feature
    'P669', # located on street

    'P36', # capital
    'P38', # currency
    'P37', # official language
    'P85', # anthem
    'P122', # head of state
    'P194', # legislative body
    'P150', # contains administrative territorial entity
    'P1383', # contains settlement
    'P1001', # applies to jurisdiction
    'P6', # head of government
    'P1336', # territory claimed by
    'P210', # party chief representative
    'P47', # shares border with
    'P500', # exclave of
    'P501', # enclave within
    'P190', # sister city
    'P84', # architect
    'P631', #  structural engineer
    'P167', # structure replaced by
    'P193', # main building contractor
    'P126', # maintained by
    'P833', # interchange station
    'P1382', # coincident with
    'P137', # operator
    'P197', # adjacent station
    'P1192', # connecting service
    'P403', # mouth of the watercourse
    'P469', # lakes on river
    'P200', # lake inflows
    'P201', # lake outflow
    'P205', # basin country
    'P206', # located next to body of water
    'P59', # constellation
    'P65', # site of astronomical discovery
    'P196', # minor planet group
    'P367', # located on astronomical body
    'P397', # astronomical body
    'P398' # child astronomical body
]
WIKIDATA_PROPERTIES_DICT = {
    # people properties
    19: 'place of birth',
    20: 'place of death',
    509: 'cause of death',
    157: 'killed by',
    119: 'place of burial',
    66: 'ancestral home',
    463: 'member of',
    172: 'ethnic group',
    103: 'native language',
    27: 'country of citizenship',
    69: 'educated at',
    106: 'occupation',
    101: 'field of work',
    800: 'notable work',
    108: 'employer',
    39: 'position held',
    102: 'member of political party',
    263: 'official residence',
    140: 'religion',
    91: 'sexual orientation',
    184: 'doctoral advisor',
    185: 'doctoral student',
    1066: 'student of',
    802: 'student',
    53: 'noble family',
    97: 'noble title',
    512: 'academic degree',
    1412: 'affiliation',
    1429: 'pet',
    22: 'father',
    25: 'mother',
    7: 'brother',
    9: 'sister',
    26: 'spouse',
    451: 'partner',
    40: 'child',
    43: 'stepfather',
    44: 'stepmother',
    1038: 'relative',
    1290: 'godparent',
    54: 'member of sports team',
    241: 'military branch',
    410: 'military rank',
    598: 'commander of',
    607: 'conflict',

    # organization properties
    159: 'headquarters location',
    452: 'industry',
    807: 'separated from',
    1056: 'product',
    457: 'foundational text',
    112: 'founder',
    740: 'location of formation',
    1387: 'political alignment',
    1408: 'licensed to broadcast to',
    1313: 'office held by head of government',
    113: 'airline hub',
    114: 'airline alliance',
    115: 'home venue',
    118: 'league',
    286: 'head coach',
    505: 'general manager',
    634: 'captain',
    822: 'mascot',
    169: 'chief executive officer',
    488: 'chairperson',
    199: 'business division',
    355: 'subsidiaries',
    1268: 'represents organisation',
    1308: 'officeholder',
    1320: 'stock exchange',
    1037: 'manager/director',
    
    # place properties
    163: 'flag',
    417: 'patron saint',
    30: 'continent',
    17: 'country',
    131: 'located in the administrative territorial entity',
    421: 'located in time zone',
    706: 'located on terrain feature',
    669: 'located on street',

    36: 'capital',
    38: 'currency',
    37: 'official language',
    85: 'anthem',
    122: 'head of state',
    194: 'legislative body',
    150: 'contains administrative territorial entity',
    1383: 'contains settlement',
    1001: 'applies to jurisdiction',
    6: 'head of government',
    1336: 'territory claimed by',
    210: 'party chief representative',
    47: 'shares border with',
    500: 'exclave of',
    501: 'enclave within',
    190: 'sister city',
    84: 'architect',
    631: ' structural engineer',
    167: 'structure replaced by',
    193: 'main building contractor',
    126: 'maintained by',
    833: 'interchange station',
    1382: 'coincident with',
    137: 'operator',
    197: 'adjacent station',
    1192: 'connecting service',
    403: 'mouth of the watercourse',
    469: 'lakes on river',
    200: 'lake inflows',
    201: 'lake outflow',
    205: 'basin country',
    206: 'located next to body of water',
    59: 'constellation',
    65: 'site of astronomical discovery',
    196: 'minor planet group',
    367: 'located on astronomical body',
    397: 'astronomical body',
    398: 'child astronomical body'
}
