from keras.models import Graph, Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import numpy as np


np.random.seed(1000)


class TrainNN(object):
    def __init__(self, input_dim=4000):
        graph = Graph()

        half_input_dim = input_dim / 2
        hidden_dim = half_input_dim / 2
        
        graph.add_input(name='in1', input_shape=(half_input_dim,))
        graph.add_input(name='in2', input_shape=(half_input_dim,))

        graph.add_node(Dense(hidden_dim, activation='sigmoid'), name='hidden1', input='in1')
        graph.add_node(Dense(hidden_dim, activation='sigmoid'), name='hidden2', input='in2')
        graph.add_node(Dense(1, activation='sigmoid'),
                       name='pre-out',
                       inputs=['hidden1', 'hidden2'],
                       merge_mode='concat')

        # for training
        graph.add_output(name='out', input='pre-out')

        self.graph = graph


    def compile(self):
        self.graph.compile(optimizer='sgd', loss={'out': 'binary_crossentropy'})


    def train(self, X, y, epochs=10000):
        _, D = X.shape

        X_in1 = X[:,:D/2]
        X_in2 = X[:,D/2:]
        
        self.graph.fit({'in1':X_in1, 'in2':X_in2, 'out': y}, nb_epoch=epochs)


    def predict(self, X):
        _, D = X.shape

        X_in1 = X[:,:D/2]
        X_in2 = X[:,D/2:]

        y_pred = self.graph.predict({'in1':X_in1, 'in2':X_in2})['out']

        return y_pred


    def save_params(self, filename):
        self.graph.save_weights(filename, overwrite=True)

    
    def load_params(self, filename):
        self.graph.load_weights(filename)


class TransformNN(object):
    def __init__(self, params_filename, input_dim=4000):
        train_nn = TrainNN(input_dim)
        train_nn.load_params(params_filename)

        half_input_dim = input_dim / 2
        hidden_dim = half_input_dim / 2
        
        weights_one = train_nn.graph.nodes['hidden1'].get_weights()
        weights_two = train_nn.graph.nodes['hidden2'].get_weights()

        del train_nn

        model_one = Sequential()
        model_one.add(Dense(hidden_dim,
                            activation='sigmoid',
                            weights=weights_one))
        # need only to compile, no training
        model_one.compile(loss='mse', optimizer='sgd')

        model_two = Sequential()
        model_two.add(Dense(hidden_dim,
                            activation='sigmoid',
                            weights=weights_two))
        # need only to compile, no training
        model_two.compile(loss='mse', optimizer='sgd')

        self.model_one = model_one
        self.model_two = model_two


    def transform(self, X, model='one'):
        if model == 'one':
            return self.model_one.predict(X)
        elif model == 'two':
            return self.model_one.predict(X)
        else:
            raise Exception("'model' param must me 'one' or 'two'")
