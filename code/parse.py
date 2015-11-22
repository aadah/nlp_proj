#!/usr/bin/env python

from bs4 import BeautifulSoup
import os  
import config
from ner_recognize import ner_recognize_string
from cluster import RelationCluster

class Article:
    
    def __init__(self, text):
        self.text = text
        # body tags cannot be accessed by BeautifulSoup
        soup = BeautifulSoup(self.text.replace("BODY","CONTENT"), 'lxml')
        self.body = soup.content

    def get_entity_set(self):
        return ner_recognize_string(self.body)

def readSgm(fname, entity_set_list):
    with open(fname, 'r') as f:
        inArticle = False
        text = ''
        for line in f:
            if inArticle:
                text += line
                if line.startswith("</REUTERS>"):
                    inArticle = False
                    a = Article(text)
                    entity_set_list.append(a.get_entity_set)
            elif line.startswith("<REUTERS"):
                inArticle = True
    return entity_set_list
                    
def readDir(dirName):
    entity_sets = []
    for fname in os.listdir(dirName):
        if fname.endswith('.sgm'):
            entity_sets += readSgm('%s/%s' %(dirName, fname), [])
    return entity_sets

def main():
    entity_sets = readDir(config.REUTERS_DIR)
    rc = RelationCluster(entity_sets)
    rc.print_clusters()
    
if __name__=="__main__":
    main()
