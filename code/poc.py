import numpy as np
import scipy.spatial as sps
from pprint import pprint

import vector


def poc_one():
    analogies = [
        ('Barack Obama', 'Michelle Obama', 'David Cameron'), # Samantha Cameron
        ('Japan', 'Tokyo', 'Germany'), # Berlin
        ('Coldplay', 'Chris Martin', '30 Seconds to Mars'), # Jared Leto
        ('Microsoft', 'Bill Gates', 'Google'), # Larry Page or Sergey Brin
        ('John F. Kennedy', 'Harvard University', 'Claude Elwood Shannon') # MIT
    ]

    model = vector.VectorModel()

    for (a, b, x) in analogies:
        a = model[a]
        b = model[b]
        x = model[x]

        y = (b-a) + x
        
        print pprint(model._most_similar(y, 'DUMMY', k=5))

    model.close()


def poc_two():
    train = [
        ('Barack Obama', 'Michelle Obama'),
        ('David Cameron', 'Samantha Cameron'),
        ('John Lennon', 'Yoko Ono'),
        ('Benjamin Millepied', 'Natalie Portman'),
        ('Bill Clinton', 'Hillary Rodham Clinton'),
        ('Will Smith', 'Jada Pinkett Smith'),
        ('David Beckham', 'Victoria Beckham'),
        ('Brad Pitt', 'Angelina Jolie'),
        ('John Osbourne', 'Sharon Osbourne'),
        ('Jay-Z', 'Beyonce')
    ]

    test = [
        ('Bill Gates', 'Melinda Gates'),
        ('Xi Jinping', 'Peng Liyuan'),
        ('John Krasinski', 'Emily Blunt'),
        ('Chris Martin', 'Gwyneth Paltrow'),
        ('Steve Jobs', 'Laurene Powell')
    ]

    model = vector.VectorModel()

    relations = [model[y]-model[x] for (x, y) in train]
    relation = sum(relations)
    relation /= len(relations)

    for (x, y) in test:
        g1 = model[x] + relation
        g2 = model[y] - relation

        # forward direction
        print '%s --> ? (%s)' % (x, y)
        pprint(model._most_similar(g1, 'DUMMY', k=5))
        print

        # backward direction
        print '? (%s) <-- %s' % (x, y)
        pprint(model._most_similar(g2, 'DUMMY', k=5))
        print

    model.close()


def main():
    #poc_one()
    poc_two()


main()
