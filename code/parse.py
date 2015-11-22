#!/usr/bin/env python

from bs4 import BeautifulSoup
import nltk
import os  
import config

STOPWORDS = nltk.corpus.stopwords.words('english')
STEMMER = nltk.stem.SnowballStemmer('english')

class Article:
    
    def __init__(self, text, meta):
        self.text = text
        # body tags cannot be accessed by BeautifulSoup
        soup = BeautifulSoup(self.text.replace("BODY","CONTENT"), 'lxml')
        '''
        self.date = soup.date
        self.topics = soup.topics
        self.places = soup.places
        self.people = soup.people
        self.orgs = soup.orgs
        self.exchanges = soup.exchanges
        self.companies = soup.companies
        self.title = soup.title
        self.dateline = soup.dateline
        '''
        self.body = soup.content
        #print self.id

    def get_entity_set(self):
        pass

    '''
    def getId(self):
        return self.id

    def removeStopwords(self, text):
        new_text = ''
        for word in text:
            if word not in STOPWORDS:
                new_text += word + ' '
        return new_text

    def stemWords(self, text):
        new_text = ''
        for word in text:
            new_text += STEMMER.stem(word) + ' '
        return new_text

    def getBagOfWords(self):
        pass
    '''

class Dossier:
    
    def __init__(self):
        self.articles = {}
    
    def addArticle(self, article):
        # Takes in an Article object and adds to its dictionary of articles
        # Articles are keyed by their id, which is the "NEWID" field in each
        # article's header
        articleId = article.getId()
        if articleId in self.articles:
            print "WARNING: ID COLLISION DETECTED FOR ID: " + articleId
            altId = articleId + '_collision'
            self.articles[altId] = article
            print "ARTICLE STORED WITH ID: " + altId
        else:
            self.articles[articleId] = article

    def getArticle(self, id):
        return articles[id]

    def readSgm(self, fname):
        with open(fname, 'r') as f:
            inArticle = False
            meta = ''
            text = ''
            for line in f:
                if inArticle:
                    text += line
                    if line.startswith("</REUTERS>"):
                        inArticle = False
                        a = Article(text, meta)
                        self.addArticle(Article(text,meta)) 
                elif line.startswith("<REUTERS"):
                    meta = line
                    # check if the article is used for the ModApte split
                    validMeta = self.readMeta(meta)
                    if validMeta:
                        inArticle = True
                    
    def readDir(self, dirName):
        for fname in os.listdir(dirName):
            if fname.endswith('.sgm'):
                return self.readSgm('%s/%s' %(dirName, fname))

    '''     
    def readMeta(self, meta):
        metaFields = [item.split('=') for item in meta.split() if len(item.split('=')) > 1]
        metaDict = {field[0].lower():field[1] for field in metaFields}
        for item in meta.split():
            field = item.replace('"','').split('=')
            if field[0] == 'TOPICS':
                self.hasTopics = (field[1] == 'YES')
            elif field[0] == 'LEWISSPLIT':
                self.lewisSplit = (field[1])
            elif field[0] == 'NEWID':
                self.id = int(field[1][:-1])
        print metaDict
        return False
    '''

def main():
    d = Dossier()
    d.readDir(config.REUTERS_DIR)
    print d.getArticle(1)
    
if __name__=="__main__":
    main()
