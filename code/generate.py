import os.path
import random
import numpy as np
from pprint import pprint

import vector
import sparql
import config


def make_instance_instance_txt():
    np.random.seed(1000)
    random.seed(1000)

    wd = sparql.WikiDataClient()
    vm = vector.MIDVectorModel()

    print 'fetching . . .'

    if os.path.isfile(config.RAW_DATA_TXT):
        with open(config.RAW_DATA_TXT) as f:
            relations_dict = eval(f.read())
    else:
        query_limit = 10000
        relations_dict = wd.get_instances(config.WIKIDATA_PROPERTIES, query_limit)
        with open(config.RAW_DATA_TXT, 'w') as f:
            pprint(relations_dict, stream=f)

    relations = relations_dict.keys()

    num_pos = 4000
    num_relations = len(relations)
    num_per_relation = num_pos / num_relations

    pos = []
    neg = []    

    print 'making positive . . .'
    for relation in relations:
        results = relations_dict[relation]
        results = map(lambda r: (r['subjLabel'],
                                 r['subjMID'],
                                 r['objLabel'],
                                 r['objMID']),
                      results)
        results = filter(lambda r: r[1] in vm and r[3] in vm,
                         results)

        print relation, len(results)

        relations_dict[relation] = results
        results_ids = range(len(results))

        for _ in xrange(num_per_relation):
            i, j = np.random.choice(results_ids, 2).tolist()
            pos_ex = (results[i], results[j], int(relation[1:]), int(relation[1:]))
            pos.append(pos_ex)

    num_neg = len(pos)

    print 'making negative . . .'
    for _ in xrange(num_neg):
        rel_i, rel_j = np.random.choice(relations, 2).tolist()
        i = np.random.randint(len(relations_dict[rel_i]))
        j = np.random.randint(len(relations_dict[rel_j]))
        neg_ex = (relations_dict[rel_i][i],
                  relations_dict[rel_j][j],
                  int(rel_i[1:]),
                  int(rel_j[1:]))
        neg.append(neg_ex)

    examples = pos + neg

    with open(config.NEW_DATA_TXT, 'w') as f:
        for ex in examples:
            f.write(repr(ex)+'\n')

    vm.close()


def make_instance_instance_npy():
    vm = vector.MIDVectorModel()
    pos = []
    neg = []

    split = 0.05

    print 'loading . . .'
    with open(config.NEW_DATA_TXT) as f:
        data = map(eval, f.readlines())

    print 'converting . . .'
    for datum in data:
        a = vm[datum[0][1]]
        b = vm[datum[0][3]]
        c = vm[datum[1][1]]
        d = vm[datum[1][3]]
        l1 = np.array([datum[2]])
        l2 = np.array([datum[3]])
        vec = np.hstack([a,b,c,d,l1,l2])

        if l1 == l2:
            pos.append(vec)
        else:
            neg.append(vec)

    print 'shuffling . . .'
    random.shuffle(pos)
    random.shuffle(neg)

    print 'building . . .'
    num_pos = len(pos)
    num_neg = len(neg)
    num_train_pos = int(num_pos*(1-split))
    num_train_neg = int(num_neg*(1-split))
    
    train_pos = pos[:num_train_pos]
    train_neg = neg[:num_train_neg]
    test_pos = pos[num_train_pos:]
    test_neg = neg[num_train_neg:]

    train = np.vstack(train_pos + train_neg)
    test = np.vstack(test_pos + test_neg)

    print 'train:', train.shape
    print 'test:', test.shape

    print 'saving . . .'
    np.save(config.NEW_DATA_TRAIN_NPY, train)
    np.save(config.NEW_DATA_TEST_NPY, test)

    vm.close()

###########################################################

# generates list of instance-instance for auto encoder
def generate_instance_to_instance():
    np.random.seed(1000)

    wd = sparql.WikiDataClient()
    limit = 10000
    num_per_query = 25
    f = open(config.AUTOENCODER_FILE, 'w')

    print 'retrieving instances from wikidata . . .'
    for relation in config.WIKIDATA_PROPERTIES:
        try:
            pre_instances = wd.get_instance(relation, limit)
        except Exception:
            print 'Error for %s' % relation
            continue

        pre_instances.sort(key=lambda inst: inst['subjLabel'])

        if len(pre_instances) == 0:
            print 'Nothing for %s' % relation
            continue

        instances = pre_instances[:num_per_query]

        for instance_i in instances:
            for instance_j in instances:
                first = (instance_i['subjLabel'],
                         instance_i['subjMID'],
                         instance_i['objLabel'],
                         instance_i['objMID'])
                
                second = (instance_j['subjLabel'],
                          instance_j['subjMID'],
                          instance_j['objLabel'],
                          instance_j['objMID'])

                instance_pair = (first, second)
                f.write(str(instance_pair) + '\n')

        del instances
        del pre_instances
        print relation

    f.close()


def generate_instance_to_instance_matrix():
    vm = vector.MIDVectorModel()

    print 'loading instance pairs . . .'
    with open(config.AUTOENCODER_FILE) as f:
        instance_pairs = map(eval, f.readlines())
        instance_pairs = map(lambda x: (x[0][1], x[0][3],
                                        x[1][1], x[1][3]),
                             instance_pairs)

    print 'building data matrix . . .'
    data = []
    
    for instance_pair in instance_pairs:
        mids = instance_pair

        if not all([mid in vm for mid in mids]):
            continue

        vec = np.hstack([vm[mid] for mid in mids])
        data.append(vec)
        
    vm.close()

    matrix = np.vstack(data)
    print matrix.shape

    np.save(config.AUTOENCODER_DATA, matrix)


# generates list of instance-instance classifier
def generate_instance_instance():
    np.random.seed(100)

    wd = sparql.WikiDataClient()

    num_per_query = 10000
    num_per_instance = 50

    print 'retrieving instances from wikidata . . .'
    results = wd.get_instances(config.WIKIDATA_PROPERTIES, num_per_query)

    instance_pairs = []
    
    print 'generating positive examples . . .'
    for relation in results:
        partial_results = results[relation]
        num = len(partial_results)
        
        for _ in xrange(num_per_instance):
            instance_i, instance_j = np.random.choice(partial_results, 2).tolist()

            first = (instance_i['subjLabel'],
                     instance_i['subjMID'],
                     instance_i['objLabel'],
                     instance_i['objMID'])
            
            second = (instance_j['subjLabel'],
                      instance_j['subjMID'],
                      instance_j['objLabel'],
                      instance_j['objMID'])
                
            instance_pairs.append((first, second, 1)) # one for positive

    print 'generating negative examples . . .'
    num_pos = len(instance_pairs)
    num_neg = 0

    while num_pos > num_neg:
        rel_i, rel_j = np.random.choice(results.keys(), 2).tolist()

        instance_i = np.random.choice(results[rel_i])
        instance_j = np.random.choice(results[rel_j])

        first = (instance_i['subjLabel'],
                 instance_i['subjMID'],
                 instance_i['objLabel'],
                 instance_i['objMID'])
        second = (instance_j['subjLabel'],
                  instance_j['subjMID'],
                  instance_j['objLabel'],
                  instance_j['objMID'])
                
        instance_pairs.append((first, second, 0)) # zero for negative
        num_neg += 1

    print 'saving to a file . . .'
    with open(config.TWO_INSTANCE_FILE, 'w') as f:
        for instance_pair in instance_pairs:
            f.write(str(instance_pair) + '\n')


def generate_instance_instance():
    np.random.seed(100)

    wd = sparql.WikiDataClient()

    num_per_query = 10000
    num_per_instance = 50

    print 'retrieving instances from wikidata . . .'
    results = wd.get_instances(config.WIKIDATA_PROPERTIES, num_per_query)

    instance_pairs = []
    
    print 'generating positive examples . . .'
    for relation in results:
        partial_results = results[relation]
        num = len(partial_results)
        
        for _ in xrange(num_per_instance):
            instance_i, instance_j = np.random.choice(partial_results, 2).tolist()

            first = (instance_i['subjLabel'],
                     instance_i['subjMID'],
                     instance_i['objLabel'],
                     instance_i['objMID'])
            
            second = (instance_j['subjLabel'],
                      instance_j['subjMID'],
                      instance_j['objLabel'],
                      instance_j['objMID'])
                
            instance_pairs.append((first, second, 1)) # one for positive

    print 'generating negative examples . . .'
    num_pos = len(instance_pairs)
    num_neg = 0

    while num_pos > num_neg:
        rel_i, rel_j = np.random.choice(results.keys(), 2).tolist()

        instance_i = np.random.choice(results[rel_i])
        instance_j = np.random.choice(results[rel_j])

        first = (instance_i['subjLabel'],
                 instance_i['subjMID'],
                 instance_i['objLabel'],
                 instance_i['objMID'])
        second = (instance_j['subjLabel'],
                  instance_j['subjMID'],
                  instance_j['objLabel'],
                  instance_j['objMID'])
                
        instance_pairs.append((first, second, 0)) # zero for negative
        num_neg += 1

    print 'saving to a file . . .'
    with open(config.TWO_INSTANCE_FILE, 'w') as f:
        for instance_pair in instance_pairs:
            f.write(str(instance_pair) + '\n')


def generate_instance_instance_matrix():
    vm = vector.MIDVectorModel()

    print 'loading instance pairs . . .'
    with open(config.TWO_INSTANCE_FILE) as f:
        instance_pairs = map(eval, f.readlines())
        instance_pairs = map(lambda x: (x[0][1], x[0][3], x[1][1], x[1][3], x[2]),
                             instance_pairs)

    print 'building data matrix . . .'
    N = len(instance_pairs)
    data = []
    
    for n in xrange(N):
        instance_pair = instance_pairs[n]
        mids = instance_pair[:-1]
        class_var = np.array([instance_pair[-1]])

        if not all([mid in vm for mid in mids]):
            continue

        vec = np.hstack([vm[mid] for mid in mids])
        data.append(np.hstack((vec, class_var)))
        
    vm.close()

    matrix = np.vstack(data)
    print matrix.shape

    np.save(config.TWO_INSTANCE_DATA, matrix)


def main():
    generate_instance_instance()
    generate_instance_instance_matrix()


def main2():
    generate_instance_to_instance()
    generate_instance_to_instance_matrix()


def main3():
    #make_instance_instance_txt()
    make_instance_instance_npy()

if __name__ == '__main__':
    #main()
    #main2()
    main3()
