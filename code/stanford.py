import nltk
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
import collections
import config

KEY_RELATIONS = ['agent','comp','obj','dobj','iobj','pobj','subj','nsubj','nsubjpass','csubj','cc','conj','appos','rcmod','ref','pobj']
STOP_ENTITIES = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday',
                 'January','February','March','April','May','June','July','August','September','November','December']
TITLES = ['Mr', 'Ms', 'Mrs', 'Prof', 'Dr']

class DepParser:
    def __init__(self):
        self.parser = StanfordDependencyParser(path_to_jar=config.STANFORD_PARSER_JAR,
                                               path_to_models_jar=config.STANFORD_PARSER_MODEL)

    def get_entity_pairs(self, text):
        pairs = []
        sents = nltk.sent_tokenize(text)
        for sent in sents:
            pairs.extend(self._get_entity_pairs(sent))
        return pairs
        
    def _get_entity_pairs(self, sent):
        #words = nltk.word_tokenize(sent)
        relations = [list(parse.triples()) for parse in self.parser.raw_parse(sent)]
        """
        print '***RELATIONS***'
        for r in relations[0]:
            print r
        """
        nnp_relations = self.filter_for_NNP(relations)

        print '***ONLY NAMED ENTITIES***'
        for r in nnp_relations:
            print r

        pairs = self.build_relation_pairs(nnp_relations, sent)
        return pairs

    def build_compound_dict(self, relations, words):
        compound_dict = collections.defaultdict(list)
        # works on the assumption that there are usually not many shared last names
        # so we can use the last name as the anchor for a compound NNP
        in_progress = False
        current = ''
        for r in relations:
            if r[1] == 'compound':
                # To prevent "Taipei, Taiwan" from being considered a compound entity
                if r[0][0] in words and words[words.index(r[0][0]) - 1] == ',':                    
                    continue
                if r[2][0] in TITLES:
                    continue
                current = r[0]
                compound_dict[r[0]].append(r[2][0])
                in_progress = True
            elif in_progress:
                in_progress = False
                if current[1] != 'NNS':
                    # We want to keep NNS entities because the compound modifiers preceding them
                    # could be important, but we don't want them being a part of set of named entities
                    compound_dict[current].append(current[0])
                current = ''
        # To catch ending compound entities
        if in_progress:
            if current[1] != 'NNS':                
                compound_dict[current].append(current[0])
        return compound_dict

    def normalize(self, entity, compound_dict):
        if entity in compound_dict:
            return ' '.join(compound_dict[entity])
        if type(entity) is tuple:
            entity = entity[0]
        return entity

    def build_relation_dict(self, relations, words):
        relation_dict = collections.defaultdict(set)
        related = set()
        for r in relations:
            if r[1] == 'compound' and r[0][0] in words:
                i = words.index(r[0][0])
                if words[i-1] == ',':
                    relation_dict[r[0]].add(r[2])
                    relation_dict[r[2]].add(r[0])
                continue
            #if r[1] in KEY_RELATIONS:
            relation_dict[r[0]].add(r[2])
            relation_dict[r[2]].add(r[0])
            related.add(r[2])
        return relation_dict

    def build_relation_pairs(self, relations, sent):
        pairs = set()
        words = nltk.word_tokenize(sent)
        relation_dict = self.build_relation_dict(relations, words)
        compound_dict = self.build_compound_dict(relations, words)
        subj = self.get_subj(relations)
        subj_norm = self.normalize(subj,compound_dict)
        obj = self.get_obj(relations)
        obj_norm = self.normalize(obj,compound_dict)
        print 'SUBJECT', subj_norm
        print 'OBJECT', obj_norm
        for entity in relation_dict:
            if not self.is_NNP(entity) or entity in STOP_ENTITIES:
                continue
            if subj and subj != entity:
                pairs.add((self.normalize(entity,compound_dict),subj_norm))
                pairs.add((subj_norm,self.normalize(entity,compound_dict)))
            if obj and obj != entity:
                pairs.add((self.normalize(entity,compound_dict),obj_norm))
                pairs.add((obj_norm,self.normalize(entity,compound_dict)))
            for one_deg_sep in relation_dict[entity]:
                if self.is_NNP(one_deg_sep):
                    if entity == one_deg_sep:
                        continue
                    pairs.add((self.normalize(entity,compound_dict),
                               self.normalize(one_deg_sep,compound_dict)))
                for two_deg_sep in relation_dict[one_deg_sep]:
                    if self.is_NNP(two_deg_sep):
                        if entity == two_deg_sep:
                            continue
                        pairs.add((self.normalize(entity,compound_dict),
                                   self.normalize(two_deg_sep,compound_dict)))
        return pairs

    def is_NNP(self, ent):
        return ent[1] in ['NNP','NNPS','NNS']

    def filter_for_NNP(self, relations):
        return [r for r in relations[0] if self.is_NNP(r[0]) or self.is_NNP(r[2])]

    def get_subj(self, relations):
        for r in relations:
            if 'subj' in r[1] or r[1] == 'agent':
                subj = r[2]
                if self.is_NNP(r[2]):
                    return r[2]
                for r in relations:
                    if r[0] == subj and self.is_NNP(r[2]):
                        return r[2]
    def get_obj(self, relations):
        for r in relations:
            if 'obj' in r[1]:
                obj = r[2]
                if self.is_NNP(r[2]):
                    return r[2]
                for r in relations:
                    if r[0] == obj and self.is_NNP(r[2]):
                        return r[2]

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

if __name__=='__main__':
    p = DepParser()
    text = 'Barack Hussein Obama II is the 44th and current President of the United States, as well as the first African American to hold the office. Born in Honolulu, Hawaii, Obama is a graduate of Columbia University and Harvard Law School, where he served as president of the Harvard Law Review.'
    text = 'But the sixty-minute session in which Facebook\'s founder was first interviewed by a journalist from Wired, then joined on stage by three mobile operators, was one big yawn, a missed opportunity. Okay, the subject of the session - Facebook\'s mission to get people in the developing world online via something called internet.org - was not, on the face of it, controversial. And Mr Zuckerberg was allowed to paint his business, in that typically happy-clappy Californian way, as motivated only by a desire to enrich more lives through an internet connection. But there are some nagging questions to be asked about internet.org. It may sound great that mobile phone users in Kenya, for instance, are getting free internet access to sites like Wikipedia, one local Kenyan news site, BBC Swahili - and, of course, Facebook. But who acts as the gatekeeper for this walled garden - and what about those other local news sites that aren\'t on the site and have to charge for access? And what price net neutrality in Africa? A two-speed internet has just been ruled out in the United States by the Federal Communications Commission. But the laudable mission of internet.org could end up creating fast lanes for those deemed worthy by Mr Zuckerberg and his lieutenants. Then there\'s the thorny question of Facebook\'s relationship with mobile phone operators. It was amusing to reflect that the three mobile companies represented on the stage could probably be gobbled up for breakfast by the hugely wealthy social network without a second thought. Mr Zuckerberg said he wanted the mobile networks\' involvement in internet.org to be recognised. Mobile networks are seeing their revenues threatened by messaging apps like WhatsApp, bought by Facebook a while back. Meanwhile, they are still much more heavily regulated - Deutsche Telekom\'s boss made a call in Barcelona for internet firms to face the same level of regulation. But on stage the mobile operators joined in the love-in, praising Mr Zuckerberg\'s campaign for increasing the flow of data across their networks. Facebook\'s founder was briefly asked about regulation. His answer, somewhat bizarrely for the chief executive of a major communications company, was that he did not understand the subject - I\'m not a regulator. He kept repeating this phrase and was allowed to laugh off the very idea that regulation was anything to do with him. The self-congratulatory session ended with little light shed on how the fractious relationship between the social network and the companies which have built the internet\'s infrastructure might develop. But some time soon - just as Google has already discovered - the regulators will come knocking. Mr Zuckerberg will need to have some answers then.'
    text = 'Germany\'s parliament has voted to send military support to the US-led coalition fighting Islamic State (IS) militants in Syria.'
    print p.get_entity_pairs(text)
