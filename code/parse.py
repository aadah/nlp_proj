#!/usr/bin/env python

from bs4 import BeautifulSoup
import os  
import config
from ner_recognize import ner_recognize_string
from cluster import RelationCluster
from vector import VectorModel
from stanford import Tagger

VM = VectorModel()

def get_entity_set(text, mode='stanford', stanford_model_num=3):
    entity_set = set()
    if mode == 'stanford':
        tagger = Tagger(stanford_model_num)
        entity_set = tagger.get_entities(text)
    else:
        entity_set = ner_recognize_string(text)
    new_entity_set = set()
    for entity in entity_set:
        if entity in VM:
            new_entity_set.add(entity)
    return new_entity_set

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

def readSgm(fname, entity_pairs):
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
                    entity_set = get_entity_set(body)
                    pair_set = get_entity_pairs(entity_set)
                    print '%d entities' % len(entity_set)
                    print '%d pairs' % len(pair_set)
                    entity_pairs = entity_pairs.union(pair_set)
                    count += 1
                    print '%d articles read' % count
                    if count == 100:
                        break
            elif line.startswith("<REUTERS"):
                inArticle = True
                text = ''
    return entity_pairs

def get_entity_pairs(entity_set):
    pair_set = set()
    for entity1 in entity_set:
        for entity2 in entity_set:
            if entity1 == entity2:
                continue
            pair_set.add((entity1, entity2))
    return pair_set

def write_entity_pairs(pair_set):
    with open(config.PAIR_FILE, 'w') as f:
        for pair in pair_set:
            f.write(str(pair) + '\n')
                    
def readDir(dirName):
    pair_set = set()
    for fname in os.listdir(dirName):
        print fname
        if fname.endswith('.sgm'):
            pair_set = readSgm('%s/%s' %(dirName, fname), pair_set)
            # for testing, only work with one sgm
            break
    write_entity_pairs(pair_set)
    return pair_set

def main():
    entity_pairs = readDir(config.REUTERS_DIR)
    rc = RelationCluster(entity_pairs)
    rc.print_clusters()
    
if __name__=="__main__":
    main()
