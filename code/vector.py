from gensim.models import Word2Vec

import config


def create_freebase_model():
    return Word2Vec.load_word2vec_format(config.FREEBASE_FILE, binary=True)
