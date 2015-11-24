import nltk
from nltk.tag import StanfordNERTagger

import config


class Tagger:
    def __init__(self, model_num):
        if model_num == 3:
            pathname = config.STANFORD_3CLASS
        elif model_num == 4:
            pathname = config.STANFORD_4CLASS
        elif model_num == 7:
            pathname = config.STANFORD_7CLASS
        else:
            raise Exception('No model for:', model_num)

        self.tagger = StanfordNERTagger(pathname, config.STANFORD_NER_JAR)


    def get_pairs(self, text):
        entities = self.get_entities(text)

        return self._get_pairs(entities)

    
    def _get_pairs(self, entities):
        pairs = []
        
        for e1 in entities:
            for e2 in entities:
                if e1 != e2:
                    pairs.append((e1, e2))

        return pairs


    def get_entities(self, text):
        entities = []
        sents = nltk.sent_tokenize(text)

        for sent in sents:
            entities.extend(self._get_entities(sent))

        entities = set(entities)

        return entities


    def _get_entities(self, sent):
        words = nltk.word_tokenize(sent)
        parse = self.tagger.tag(words)
        entities = self._collapse_entities(parse)

        return entities

        
    def _collapse_entities(self, parse):
        entities = []
        prev_token = None
        prev_entity = u'O'

        for (token, entity) in parse:
            if entity == u'O':
                if prev_entity != u'O':
                    entities.append(prev_token)
                    prev_token = token
                    prev_entity = entity
            else:
                if prev_entity == u'O':
                    prev_token = token
                    prev_entity = entity
                elif prev_entity == entity:
                    prev_token += ' %s' % token
                else:
                    entities.append(prev_token)
                    prev_token = token
                    prev_entity = entity
        
        return entities
