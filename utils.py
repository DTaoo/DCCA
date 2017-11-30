# The utils is written by Vahid Noroozi, https://github.com/VahidooX/DeepCCA

import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def model_freeze_layer(model, layer_idx, freezeOrNot):
    if freezeOrNot:
        for i in range(len(layer_idx)):
            model.layers[layer_idx[i]].trainable = False
    else:
        layer_num = len(model.layers)
        for i in range(layer_num):
            if i not in layer_idx:
                model.layers[i].trainable = False
    return model


def model_free_layer(model):
    layer_num = len(model.layers)
    for i in range(layer_num):
        model.layers[i].trainable = True

    return model





