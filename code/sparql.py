import bs4
import urllib2
import re
import collections
from SPARQLWrapper import SPARQLWrapper, JSON

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

        self.organization_properties = [
            'P169', # cheif executive officer
            'P488', # chairperson
            'P112', # founder
            'P159', # headquarters location
            'P355', #subsidiaries
        ]

        self.location_properties = [
            
        ]


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
        
        results = self.client.query().convert()
        
        return results


    def _format_query(self, query):
        return '%s\n\n%s' % (self.prefixes, query)


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
