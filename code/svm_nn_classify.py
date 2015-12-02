import sklearn.svm
import numpy as np

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


def cross_validate_svc(train_set, train_labels, cv_set, cv_labels):
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
            my_svc = sklearn.svm.SVC(C=10**c_exp, kernel='rbf', degree=3,
                             gamma=10**gamma_exp, coef0=0.0, shrinking=True,
                             probability=False, tol=0.001, cache_size=200,
                             class_weight=None, verbose=False, max_iter=-1,
                             decision_function_shape=None, random_state=None)
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

def cross_validate_nn():
    return

    

if __name__ == "__main__":

    np.set_printoptions(precision=4)

    train_set, train_labels, \
               cv_set, cv_labels, \
               test_set, test_labels = get_data()

    best_c, best_gamma = cross_validate_svc(train_set, train_labels, cv_set, cv_labels)

        
