import bs4
import urllib2
import re
import collections
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np

import mappings
import vector
import config


class WikiDataClient:
    def __init__(self):
        self.endpoint = config.WIKIDATA_ENDPOINT
        self.client = SPARQLWrapper(config.WIKIDATA_ENDPOINT)
        self.prefixes = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX v: <http://www.wikidata.org/prop/statement/>
PREFIX q: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"""


    def get_entity(self, query_string):
        rgx = re.compile('\s+')
        query_string = rgx.sub('+', query_string)
        query_url = config.WIKIDATA_QUERY_TEMPLATE % (query_string, query_string)

        try:
            result = urllib2.urlopen(query_url).read()
        except:
            result = ''

        obj = bs4.BeautifulSoup(result)
        divs = obj.findAll('div', {'class': "mw-search-result-heading"})

        if len(divs) == 0:
            return None

        div = divs[0]
        q = div.a.attrs['href'].split('/')[-1]

        return q


    def get_instances(self, relations, number_per_relation):
        results_dict = {}
        try_later = []
        try_again_later = []

        for relation in relations:
            try:
                results = self.get_instance(relation, number_per_relation)
            except Exception:
                print relation, '.'
                try_later.append(relation)
                continue

            if len(results) > 1:
                results_dict[relation] = results
            else:
                print 'No results for', relation

        for relation in try_later:
            try:
                results = self.get_instance(relation, number_per_relation)
            except Exception:
                print relation, '. .'
                try_again_later.append(relation)
                continue

            if len(results) > 1:
                results_dict[relation] = results
            else:
                print 'No results for', relation

        for relation in try_again_later:
            try:
                results = self.get_instance(relation, number_per_relation)
            except Exception:
                print relation, '. . .'
                continue

            if len(results) > 1:
                results_dict[relation] = results
            else:
                print 'No results for', relation

        return results_dict


    def get_instance(self, relation, number):
        query = """SELECT ?subjLabel ?objLabel ?subjMID ?objMID WHERE {
   ?subj wdt:%s ?obj .
   ?subj wdt:P646 ?subjMID .
   ?obj wdt:P646 ?objMID .
   
   SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
   }
 } LIMIT %d"""

        query = query % (relation, number)
        bindings = self._query(query)['results']['bindings']
        results = process_bindings(bindings)

        return results

    
    def relation_histo(self, cluster):
        histo = collections.defaultdict(int)
        all_relations = map(self.relations, cluster)

        for relations in all_relations:
            if len(relations) == 0:
                histo['NOT_RELATED'] += 1
                continue

            for relation in relations:
                histo[relation] += 1

        return histo
        

    def relations(self, pair):
        # convert to MIDs here
        #mids = pair

        mid1 = mappings.convert_to_freebase_mid(vector.format_string(pair[0]))
        mid2 = mappings.convert_to_freebase_mid(vector.format_string(pair[1]))
        mids = (mid1, mid2)
        
        return self._relations(mids)


    def _relations(self, mids):
        query = """SELECT * WHERE {
    ?subj ?rel ?obj .
    ?subj wdt:P646 "%s" .
    ?obj wdt:P646 "%s" .
}"""

        mid1 = mids[0]
        mid2 = mids[1]

        results = self._query(query % (mid1, mid2))['results']['bindings']
        
        return map(lambda x: x['rel']['value'], results)

        
    def _query(self, query):
        query = self._format_query(query)
        self.client.setQuery(query)
        self.client.setReturnFormat(JSON)
        
        q = self.client.query()
        results = q.convert()
        
        return results


    def _format_query(self, query):
        return '%s\n\n%s' % (self.prefixes, query)

#########################################################

def load_relation_dict_from_file():
    wd = WikiDataClient()
    relation_dict = {}

    with open(config.TWO_INSTANCE_FILE) as f:
        tuples = map(eval, f.readlines())

    for p1, p2, _ in tuples:
        mids1 = (p1[1],p1[3])
        mids2 = (p2[1],p2[3])

        rel1 = wd._relations(mids1)
        rel2 = wd._relations(mids2)

        if len(rel1) > 0:
            rel1 = rel1[0]
            
            if rel1 in relation_dict:
                relation_dict[rel1].add(mids1)
            else:
                relation_dict[rel1] = set([mids1])

        if len(rel2) > 0:
            rel2 = rel2[0]
            
            if rel2 in relation_dict:
                relation_dict[rel2].add(mids2)
            else:
                relation_dict[rel2] = set([mids2])

    return relation_dict


def convert_rel_dict_to_matricies():
    vm = vector.MIDVectorModel()
    
    with open(config.RELATIONS_DICT) as f:
        relations_dict = eval(f.read())
        
    relations = relations_dict.keys()
    mapping = {r: i for (i, r) in enumerate(relations)}
    split = 0.05
    train = []
    test = []
    
    for relation in relations:
        instances = relations_dict[relation]
        instances = map(lambda inst: (inst['subjMID'], inst['objMID']), instances)
        instances = filter(lambda inst: inst[0] in vm and inst[1] in vm, instances)

        l = len(instances)
        i = int(l*(1-split))

        train.append((instances[:i], mapping[relation]))
        test.append((instances[i:], mapping[relation]))

    train_vecs = []
    test_vecs = []

    print 'create training data'
    for instances, label in train:
        for mid1, mid2 in instances:
            vec = np.hstack([vm[mid1], vm[mid2], np.array([label])])
            train_vecs.append(vec)

    print 'create testing data'
    for instances, label in test:
        for mid1, mid2 in instances:
            vec = np.hstack([vm[mid1], vm[mid2], np.array([label])])
            test_vecs.append(vec)

    train_matrix = np.vstack(train_vecs)
    test_matrix = np.vstack(test_vecs)

    np.save(config.SINGLE_INSTANCE_TRAIN, train_matrix)
    np.save(config.SINGLE_INSTANCE_TEST, test_matrix)

    vm.close()


#########################################################

def get_familial_relations(wd, num_per_rel):
    person_properties = [
        'P22', # father
        'P25', # mother
        'P7', # brother
        'P9', # sister
        'P26', # spouse
        'P451', # partner
        'P40', # child
        #'P43', # stepfather (too few)
        #'P44', # stepmother (too few)
        'P1038', # relative
        #'P1290' # godparent (too few)
    ]

    pos_query = """SELECT ?subjLabel ?objLabel ?subjMID ?objMID WHERE {
   ?subj wdt:%s ?obj .
   ?subj wdt:P31 wd:Q5 .
   ?obj wdt:P31 wd:Q5 .
   ?subj wdt:P646 ?subjMID .
   ?obj wdt:P646 ?objMID .
   
   SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
   }
 } LIMIT %d"""

    neg_query = """SELECT ?subjLabel ?objLabel ?subjMID ?objMID WHERE {
   ?subj wdt:P31 wd:Q5 .
   ?obj wdt:P31 wd:Q5 .
   ?subj wdt:P646 ?subjMID .
   ?obj wdt:P646 ?objMID .
  
   FILTER NOT EXISTS {
       %s
   }
   
   SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
   }
 } LIMIT %d"""

    all_bindings = []

    for rel in person_properties:
       query = pos_query % (rel, num_per_rel)
       bindings = wd._query(query)['results']['bindings']
       all_bindings.append(bindings)
       
    #statements = '\n'.join(map(lambda s: '?subj wdt:%s ?obj .' % s, person_properties))
    #bindings = wd._query(neg_query % (statements, num_per_rel))['results']['bindings']
    #all_bindings.append(bindings)
    #person_properties.append('NOT_RELATED')

    return {cls: process_bindings(bindings) for cls, bindings in zip(person_properties, all_bindings)}


def process_bindings(bindings):
    new_bindings = []

    for binding in bindings:
        new_binding = {key: binding[key]['value'] for key in binding}
        new_bindings.append(new_binding)

    return new_bindings


def main2():
    wd = WikiDataClient()
    #ans = wd._relations(('/m/02mjmr', '/m/025s5v9'))
    ans = wd.relations(('Coldplay', 'Chris Martin'))

    return ans


def main():
    wd = WikiDataClient()
    results = get_familial_relations(wd, 200)

    return results


if __name__ == '__main__':
    pass
    #main()
