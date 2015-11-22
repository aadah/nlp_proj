#!/usr/bin/env python

from bs4 import BeautifulSoup
import os  
import config
from ner_recognize import ner_recognize_string
from cluster import RelationCluster

def get_entity_set(text):
    return ner_recognize_string(text)

def get_article_text(text):
    soup = BeautifulSoup(text.replace("BODY","CONTENT"), 'lxml')
    if soup.content is not None:
        return soup.content.getText()

def readSgm(fname, entity_set_list):
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
                    entity_set = get_entity_set(body)
                    print entity_set
                    entity_set_list.append(entity_set)
                    count += 1
                    print '%d articles read' % count
            elif line.startswith("<REUTERS"):
                inArticle = True
                text = ''
    return entity_set_list
                    
def readDir(dirName):
    entity_sets = []
    for fname in os.listdir(dirName):
        print fname
        if fname.endswith('.sgm'):
            entity_sets += readSgm('%s/%s' %(dirName, fname), [])
            print entity_sets
            # for testing, only work with one sgm
            break
    return entity_sets

def main():
    entity_sets = readDir(config.REUTERS_DIR)
    rc = RelationCluster(entity_sets)
    rc.print_clusters()
    
if __name__=="__main__":
    main()
