import numpy as np

import vector
import sparql
import config


# generates list of instance-instance classifier
def generate_instance_instance():
    np.random.seed(100)

    vm = vector.MIDVectorModel()
    wd = sparql.WikiDataClient()

    num_per_query = 1000
    num_per_instance = 30

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


if __name__ == '__main__':
    main()
