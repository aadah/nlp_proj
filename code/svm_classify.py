import sklearn.svm
import numpy as np
import math
import sys

import config

from sklearn.externals import joblib

def get_random_data(train_size=1000, cv_size=100, test_size=100):
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
           all_data[train_size:train_size+cv_size,:num_features], all_labels[train_size:train_size+cv_size], \
           all_data[train_size+cv_size:,:num_features], all_labels[train_size+cv_size:]

def get_data(train_frac=0.8, cv_frac=0.1, test_frac=0.1):

    if not (train_frac + cv_frac + test_frac == 1):
        raise ValueError('Data set fractions must add up to 1.')
    
    all_data_with_labels = np.load(config.TWO_INSTANCE_DATA)
    np.random.shuffle(all_data_with_labels)

    all_data_size = all_data_with_labels.shape[0]
    train_size = math.floor(all_data_size*train_frac)
    if (test_frac == 0):
        cv_size = all_data_size - train_size
        test_size = 0
    else:
        cv_size = math.floor(all_data_size*cv_frac)
        test_size = all_data_size - (train_size + cv_size)

    print "split fractions:", train_frac, cv_frac, test_frac
    print "split numbers:", train_size, cv_size, test_size

    all_data = all_data_with_labels[:,:-1]
    all_labels = all_data_with_labels[:,-1:]
    all_labels[all_labels == 0] = -1
    all_labels = all_labels.ravel()

    print all_labels
    print all_data

    return all_data[:train_size,:], all_labels[:train_size], \
           all_data[train_size:train_size+cv_size,:], all_labels[train_size:train_size+cv_size], \
           all_data[train_size+cv_size:,:], all_labels[train_size+cv_size:]

def get_split_data(train_frac=0.9):
    
    train_data_with_labels = np.load(config.SUBTRACT_DATA_TRAIN)
    test_data_with_labels = np.load(config.SUBTRACT_DATA_TEST)

    print "train data size:", train_data_with_labels.shape[0]
    print "test data size:", test_data_with_labels.shape[0]

    all_train_data_size = train_data_with_labels.shape[0]
    train_size = math.floor(all_train_data_size*train_frac)

    print "train fraction:", train_frac, "out of", all_train_data_size

    all_train_data = train_data_with_labels[:,:-1]
    all_train_labels = train_data_with_labels[:,-1:]
    all_train_labels[all_train_labels == 0] = -1
    all_train_labels = all_train_labels.ravel()

    test_data = test_data_with_labels[:,:-1]
    test_labels = test_data_with_labels[:,-1:]
    test_labels[test_labels == 0] = -1
    test_labels = test_labels.ravel()


    return all_train_data[:train_size,:], all_train_labels[:train_size], \
           all_train_data[train_size:,:], all_train_labels[train_size:], \
           test_data, test_labels
    

def create_svm(my_c, my_gamma):

    return sklearn.svm.SVC(C=my_c, kernel='rbf', degree=3,
                             gamma=my_gamma, coef0=0.0, shrinking=True,
                             probability=False, tol=0.001, cache_size=200,
                             class_weight=None, verbose=False, max_iter=-1,
                             decision_function_shape=None, random_state=None) 

def cross_validate_svm(train_set, train_labels, cv_set, cv_labels, exp_name):
        
    # grid search over c and gamma
    print "beginning cross-validation"
    c_exp_range = range(-8,9) #range(2,5) #range(-2,9) #range(-5,16,2)
    gamma_exp_range = range(-8,9) #range(-2,1) #range(-8,3) #range(-15,4,2)
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
    np.save(config.SVM_RESULTS_DIR+exp_name+"_cv_results.npy", results)
    best_coords = np.unravel_index(results.argmax(), results.shape)
    print best_coords
    best_c = 10**c_exp_range[best_coords[0]]
    best_gamma = 10**gamma_exp_range[best_coords[1]]
    print "best c:", best_c, ". best gamma:", best_gamma
    np.save(config.SVM_RESULTS_DIR+exp_name+"_best_c_gamma.npy", np.array([best_c, best_gamma]))
    return best_c, best_gamma


def make_and_save_svm(exp_name, train=0.9, cv=0.1, test=0.0):

    train_set, train_labels, \
        cv_set, cv_labels, \
        test_set, test_labels = get_split_data(train) #get_data(train,cv,test)

##    print "train_set shape:", train_set.shape, \
##          "train_labels shape:", train_labels.shape, \
##          "cv_set shape:", cv_set.shape, \
##          "cv_labels shape:", cv_labels.shape, \
##          "test_set shape:", test_set.shape, \
##          "test_labels shape:", test_labels.shape

    best_c, best_gamma = cross_validate_svm(train_set, train_labels,
                                            cv_set, cv_labels, exp_name)

    my_svc = create_svm(best_c, best_gamma)
    my_svc.fit(train_set, train_labels)
    if train_set.shape[0] > 0:
        score = my_svc.score(test_set, test_labels)
        labels = my_svc.predict(test_set)
        np.save(config.SVM_RESULTS_DIR+exp_name+"_test_pred.npy", labels)
        print "FINAL SCORE:", score
        
    joblib.dump(my_svc, config.SVM_RESULTS_DIR+exp_name+"_svm_model.pkl")

def load_svm(filename):
    return joblib.load(filename)

def fit_svm(test_data, filename):
    svm = load_svm(filename)
    labels = svm.fit(test_data)
    return labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "No action specified."
    elif sys.argv[1] == "cv":
        if len(sys.argv) == 6:
            make_and_save_svm(sys.argv[2], float(sys.argv[3]),
                              float(sys.argv[4]), float(sys.argv[5]))
        elif len(sys.argv) == 3:
            make_and_save_svm(sys.argv[2])
        else:
            print "Wrong number of arguments."
            print "Required arguments: experiment_name train_fraction cv_fraction test_fraction"        
    else:
        print "Command '"+sys.argv[1]+"' not recognized. Recognized commands: cv"
