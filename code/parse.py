#!/usr/bin/env python

from bs4 import BeautifulSoup
import os
import requests
import config
from ner_recognize import ner_recognize_string
from cluster import RelationCluster
from vector import VectorModel
from stanford import Tagger
from stanford import DepParser

VM = VectorModel()

def get_entity_set(text):
    entity_set = set()
    mode = config.NER_MODE
    if mode == 'stanford':
        tagger = Tagger(config.STANFORD_MODEL_NUM)
        entity_set = tagger.get_entities(text)
    else:
        entity_set = ner_recognize_string(text)
    new_entity_set = set()
    for entity in entity_set:
        if entity in VM:
            new_entity_set.add(entity)
    return new_entity_set

def get_entity_pairs(text):
    if config.PARSER == 'stanford_dep_parser':
        parser = DepParser()
        pair_set = parser.get_entity_pairs(text)
        return pair_set
    entity_set = get_entity_set(text)
    pair_set = set()
    for entity1 in entity_set:
        for entity2 in entity_set:
            if entity1 == entity2:
                continue
            pair_set.add((entity1, entity2))
    return pair_set

def write_entity_pairs(pair_set):
    with open(config.PAIR_OUTPUT_FILE, 'w') as f:        
        for pair in pair_set:
            f.write(str(pair) + '\n')
        print 'wrote %d pairs' % len(pair_set)

def read_entity_pairs():
    pair_set = set()
    with open(config.PAIR_INPUT_FILE, 'r') as f:
        for line in f:            
            pair_set.add(eval(line.strip()))
    print '%d pairs read' % len(pair_set)
    return pair_set

# Reuters specific methods
def get_article_text(text):
    soup = BeautifulSoup(text.replace("BODY","CONTENT"), 'lxml')
    if soup.content is not None:
        return soup.content.getText()
    return ''

def get_article_ents(text):
    soup = BeautifulSoup(text.replace("BODY","CONTENT"), 'lxml')
    entity_string = ''
    entity_string += soup.places.getText().upper()
    entity_string += soup.people.getText().upper()
    entity_string += soup.orgs.getText().upper()
    entity_string += soup.exchanges.getText().upper()
    entity_string += soup.companies.getText().upper()
    return entity_string

def read_sgm(fname, entity_pairs):
    count = 0
    with open(fname, 'r') as f:
        inArticle = False
        text = ''
        for line in f:
            if inArticle:
                text += line
                if line.startswith("</REUTERS>"):
                    inArticle = False
                    body = get_article_text(text)
                    print type(body)                    
                    pair_set = get_entity_pairs(body)
                    print '%d pairs' % len(pair_set)
                    print pair_set
                    entity_pairs = entity_pairs.union(pair_set)
                    count += 1
                    print '%d articles read' % count
                    if count == 100:
                        break
            elif line.startswith("<REUTERS"):
                inArticle = True
                text = ''
    return entity_pairs

def read_reuters_dir():    
    pair_set = set()
    dirName = config.REUTERS_DIR
    for fname in os.listdir(dirName):
        print fname
        if fname.endswith('.sgm'):
            pair_set = read_sgm('%s/%s' %(dirName, fname), pair_set)
            # for testing, only work with one sgm
            break
    write_entity_pairs(pair_set)
    return pair_set

# BBC specific methods
def get_linked_article(link):
    r = None
    try:
        r = requests.get(link.strip())
    except requests.exceptions.RequestException as e:
        print e
        return ''
    print r.status_code
    if r.status_code == 200:
        text = ''
        all_text = r.text
        soup = BeautifulSoup(all_text, 'lxml')
        for para in soup.find_all('p'):
            if para.get('class') is None:
                text += para.getText() + ' '
        #print text
        return text
    else:
        return ''
             
def read_links(fname, entity_pairs):
    with open(fname, 'r') as f:
        count = 0
        for line in f:
            print line.strip()
            body = get_linked_article(line)
            pair_set = get_entity_pairs(body)
            print '%d pairs' % len(pair_set)
            print pair_set
            entity_pairs = entity_pairs.union(pair_set)
            count += 1
            print '%d links read' % count
            if count == 100:
                break
    return entity_pairs

def read_bbc_dir():
    pair_set = set()
    '''
    dirName = config.BBC_DIR
    for fname in os.listdir(dirName):
        print fname
        pair_set = read_links('%s/%s' %(dirName, fname), pair_set)
    '''
    # because we moved links into code/ directory
    pair_set = read_links(config.BBC_ARTICLES, pair_set)
    write_entity_pairs(pair_set)
    return pair_set

def main():
    entity_pairs = set()
    if config.READ_FROM_PAIRS:
        entity_pairs = read_entity_pairs()
    else:
        if config.ARTICLE_SOURCE == 'reuters':
            entity_pairs = read_reuters_dir()
        elif config.ARTICLE_SOURCE == 'bbc':
            entity_pairs = read_bbc_dir()
    rc = RelationCluster(entity_pairs,
                         eps=config.CLUSTER_EPS, 
                         min_samples=config.CLUSTER_MIN_SAMPLES,
                         metric=config.CLUSTER_METRIC,
                         algorithm=config.CLUSTER_ALGO)
    rc.print_clusters()
    return rc
    
if __name__=="__main__":
    main()
