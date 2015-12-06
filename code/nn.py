from keras.models import Graph, Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np

import config


np.random.seed(1000)


def sigmoid(z):
    return np.clip(1.0 / (1 + np.exp(-z)), 1e-5, 0.99999)


class AutoEncoderNN(object):
    def __init__(self, input_dim=2000):
        model = Sequential()

        hidden_dim = input_dim / 2
        
        model.add(Dense(output_dim=hidden_dim,
                        input_dim=input_dim,
                        activation='sigmoid'))

        model.add(Dense(output_dim=input_dim,
                        input_dim=hidden_dim,
                        activation='linear'))

        self.model = model


    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse')


    def train(self, X, Y, epochs=1000):        
        self.model.fit(X, Y,
                       nb_epoch=epochs,
                       validation_split=0.05)


    def predict(self, X):
        Y_pred = self.model.predict(X)

        return Y_pred


    def save_params(self, filename):
        self.graph.save_weights(filename, overwrite=True)

    
    def load_params(self, filename):
        self.graph.load_weights(filename)


class TrainNN2(object):
    def __init__(self, input_dim=4000):
        graph = Graph()

        hidden_dim = input_dim / 2
        
        graph.add_input(name='X', input_shape=(input_dim,))

        graph.add_node(Dense(hidden_dim, activation='sigmoid'), name='hidden', input='X')

        graph.add_node(Dense(1, activation='sigmoid'),
                       name='pre-out',
                       input='hidden')

        # for training
        graph.add_output(name='out', input='pre-out')

        self.graph = graph


    def compile(self):
        self.graph.compile(optimizer='sgd', loss={'out': 'binary_crossentropy'})


    def train(self, X, y, epochs=1000):        
        self.graph.fit({'X': X, 'out': y}, nb_epoch=epochs)


    def predict_probs(self, X):
        y_pred = self.graph.predict({'X': X})['out']
        y_pred = y_pred.reshape((len(y_pred),))

        return y_pred


    def predict(self, X):
        N, _ = X.shape
        y_pred = self.predict_probs(X)

        for n in xrange(N):
            if y_pred[n] > 0.5:
                y_pred[n] = 1.0
            else:
                y_pred[n] = 0.0

        return y_pred


    def error_rate(self, X, y):
        total = y.shape[0]
        y_pred = self.predict(X)

        comp = y_pred == y
        correct = len(comp[comp==False])

        return float(correct) / total


    def save_params(self, filename):
        self.graph.save_weights(filename, overwrite=True)

    
    def load_params(self, filename):
        self.graph.load_weights(filename)


class TrainNN(object):
    def __init__(self, input_dim=4000):
        graph = Graph()

        half_input_dim = input_dim / 2
        hidden_dim = half_input_dim / 2
        
        graph.add_input(name='in1', input_shape=(half_input_dim,))
        graph.add_input(name='in2', input_shape=(half_input_dim,))


        #raph.add_node(Dense(hidden_dim, activation='sigmoid'), name='pre_hidden1', input='in1')
        #graph.add_node(Dense(hidden_dim, activation='sigmoid'), name='hidden1', input='pre_hidden1')
        #graph.add_node(Dense(hidden_dim, activation='sigmoid'), name='pre_hidden2', input='in2')
        #graph.add_node(Dense(hidden_dim, activation='sigmoid'), name='hidden2', input='pre_hidden2')

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


    def train(self, X, y, epochs=1000):
        _, D = X.shape

        X_in1 = X[:,:D/2]
        X_in2 = X[:,D/2:]
        
        self.graph.fit({'in1':X_in1, 'in2':X_in2, 'out': y}, nb_epoch=epochs)


    def predict_probs(self, X):
        _, D = X.shape

        X_in1 = X[:,:D/2]
        X_in2 = X[:,D/2:]

        y_pred = self.graph.predict({'in1':X_in1, 'in2':X_in2})['out']
        y_pred = y_pred.reshape((len(y_pred),))

        return y_pred


    def predict(self, X):
        N, _ = X.shape
        y_pred = self.predict_probs(X)

        for n in xrange(N):
            if y_pred[n] > 0.5:
                y_pred[n] = 1.0
            else:
                y_pred[n] = 0.0

        return y_pred


    def error_rate(self, X, y):
        total = y.shape[0]
        y_pred = self.predict(X)

        comp = y_pred == y
        correct = len(comp[comp==False])

        return float(correct) / total


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
        #model_one.compile(loss='mse', optimizer='sgd')

        model_two = Sequential()
        model_two.add(Dense(hidden_dim,
                            activation='sigmoid',
                            weights=weights_two))
        # need only to compile, no training
        #model_two.compile(loss='mse', optimizer='sgd')

        #self.model_one = model_one
        #self.model_two = model_two
        self.weights_one = weights_one
        self.weights_two = weights_two


    def transform(self, X, model='one'):
        if model == 'one':
            W = self.weights_one[0]
            b = self.weights_one[-1]
            return sigmoid(np.dot(X, W) + b)
            #return self.model_one.predict(X)
        elif model == 'two':
            W = self.weights_two[0]
            b = self.weights_two[-1]
            return sigmoid(np.dot(X, W) + b)
            #return self.model_one.predict(X)
        else:
            raise Exception("'model' param must be 'one' or 'two'")


def main():
    return

    data = np.load(config.TWO_INSTANCE_DATA)
    X = data[:,:-1]
    y = data[:,-1]
    _, D = X.shape

    print 'building/compiling model . . .'
    n = TrainNN(D)
    n.compile()

    print 'training . . .'
    n.train(X, y, epochs=1000)

    print 'saving model params . . .'
    n.save_params(config.TWO_INSTANCE_PARAMS)

    print 'Error rate:', n.error_rate(X, y)

    print 'Done!'


def main2():
    return

    data = np.load(config.TWO_INSTANCE_DATA)
    X = data[:,:-1]
    y = data[:,-1]
    _, D = X.shape

    print 'building/compiling model . . .'
    n = TrainNN2(D)
    n.compile()

    print 'training . . .'
    n.train(X, y, epochs=1000)

    print 'saving model params . . .'
    n.save_params(config.TWO_INSTANCE_PARAMS_CONNECT)

    print 'Error rate:', n.error_rate(X, y)

    print 'Done!'


def main3():
    #return

    data = np.load(config.AUTOENCODER_DATA)
    _, DD = data.shape
    D = DD / 2
    X = data[:,:D]
    Y = data[:,D:]

    _, D = X.shape

    print 'building/compiling model . . .'
    n = AutoEncoderNN(D)
    n.compile()

    print 'training . . .'
    n.train(X, Y, epochs=1000)

    print 'saving model params . . .'
    n.save_params(config.AUTOENCODER_PARAMS)

    print 'Done!'


if __name__ == '__main__':
    #main()
    #main2()
    main3()
