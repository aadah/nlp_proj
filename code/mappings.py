import urllib2
import re
import multiprocessing as mp
import sys


FREEBASE_URL = 'http://www.freebase.com'


def populate_queue(q, data):
    for datum in data:
        q.put(datum)


def create_queue(data=[]):
    q = mp.JoinableQueue()
    populate_queue(q, data)
    return q


class Worker:
    def __init__(self):
        self.process = None


    def work(self):
        pass


    def start(self):
        self.process = mp.Process(target=self.work)
        self.process.daemon = True
        self.process.start()


    def join(self):
        self.process.join()


def convert_to_freebase_mid(string):
    url = FREEBASE_URL + '/en/%s' % string

    try:
        result = urllib2.urlopen(url).read()
    except:
        result = ''

    mid = extract_mid(result)

    if mid:
        return mid

    return 'MISSING'


def extract_mid(result):
    rgx = re.compile('SERVER.c.id = "(/m/.+)"')
    m = rgx.search(result)
    
    if m is None: return False

    return m.group(1)


class MapperWorker(Worker):
    def __init__(self, in_queue, out_queue):
        Worker.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

    
    def work(self):
        while True:
            entity = self.in_queue.get().strip()
            mid = convert_to_freebase_mid(entity)
            ent_mid = '%s %s' % (entity, mid)

            self.out_queue.put(ent_mid)
            self.in_queue.task_done()


def main():
    with open('entities') as f:
        ents = f.readlines()
        
    num_of_workers = 8
    in_queue = create_queue(ents)
    out_queue = create_queue()
    workers = [MapperWorker(in_queue, out_queue) for i in xrange(num_of_workers)]

    for worker in workers:
        worker.start()

    i = 0
    while True:
        i += 1
        ent_mid = out_queue.get()
        print i, ent_mid
        sys.stdout.flush()
        out_queue.task_done()


if __name__ == '__main__':
    main()
