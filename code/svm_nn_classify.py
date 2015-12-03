import sklearn.svm
import numpy as np

from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

def get_data(train_size=100, cv_size=50, test_size=50):
##    print "starting to load data"
##    all_data = np.loadtxt("dummy_data/gisette_train.data.txt")
##    all_labels = np.loadtxt("dummy_data/gisette_train.labels.txt")
##    print "finishing loading data"
    num_features = 4000
    num_samples = train_size + cv_size + test_size
    all_data = np.random.rand(num_samples, num_features)
    all_labels = np.random.rand(num_samples,)
    all_labels = np.round(all_labels)

    return all_data[:train_size,:num_features], all_labels[:train_size], \
           all_data[train_size:train_size+test_size,:num_features], all_labels[train_size:train_size+test_size], \
           all_data[train_size+test_size:,:num_features], all_labels[train_size+test_size:]

def create_svm(my_c, my_gamma):

    return sklearn.svm.SVC(C=my_c, kernel='rbf', degree=3,
                             gamma=my_gamma, coef0=0.0, shrinking=True,
                             probability=False, tol=0.001, cache_size=200,
                             class_weight=None, verbose=False, max_iter=-1,
                             decision_function_shape=None, random_state=None) 

def cross_validate_svm(train_set, train_labels, cv_set, cv_labels):
    # grid search over c and gamma
    print "beginning cross-validation"
    c_exp_range = range(-3,4) #range(-5,16,2)
    gamma_exp_range = range(-3,4) #range(-15,4,2)
    results = np.zeros((len(c_exp_range),len(gamma_exp_range)))
    c_iter = 0
    gamma_iter = 0
    my_iter = 0
    for c_exp in c_exp_range:
        for gamma_exp in gamma_exp_range:
            my_svc = create_svm(10**c_exp, 10**gamma_exp)
            my_svc.fit(train_set, train_labels)
            result = my_svc.score(cv_set, cv_labels)
            results[c_iter, gamma_iter] = result
            gamma_iter+=1
        c_iter+=1
        print c_iter
        gamma_iter=0
    print results
    best_coords = np.unravel_index(results.argmax(), results.shape)
    print best_coords
    best_c = 10**c_exp_range[best_coords[0]]
    best_gamma = 10**gamma_exp_range[best_coords[1]]
    print "best c:", best_c, ". best gamma:", best_gamma
    return best_c, best_gamma

def create_nn(train_data, train_labels, test_data, test_labels):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, input_dim=train_data.shape[0], init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, init='uniform'))
    model.add(Activation('softmax'))

 #  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='sgd')
    print "compiled model"
    model.fit(train_data, train_labels, nb_epoch=20, batch_size=16)
    score = model.evaluate(test_data, test_labels, batch_size=16)
    print score


def make_and_save_svm(filename):

    train_set, train_labels, \
        cv_set, cv_labels, \
        test_set, test_labels = get_data()

    best_c, best_gamma = cross_validate_svm(train_set, train_labels, cv_set, cv_labels)

    my_svc = create_svm(best_c, best_gamma)
    my_svc.fit(train_set, train_labels)
    joblib.dump(my_svc, filename)

def load_svm(filename):
    return joblib.load(filename)

def fit_svm(test_data, filename):
    svm = load_svm(filename)
    labels = svm.fit(test_data)
    return labels
        
