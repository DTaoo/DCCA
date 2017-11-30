import sys
sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/h5py')
sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/Keras-2.0.6-py2.7.egg')

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import numpy as np
from utils import load_data, svm_classify
from keras.models import Model
import keras.backend as K
from keras.layers import Merge, Input, Dense, concatenate, Dropout
from cca_layer import CCA

def constant_loss(y_true, y_pred):
    return y_pred

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

if __name__ == '__main__':

    # size of the input for view 1 and view 2
    input_shape1 = 784
    input_shape2 = 784

    # network settings
    epoch_num = 100
    batch_size = 1000

    #load data
    data1 = load_data('noisymnist_view1.gz')
    data2 = load_data('noisymnist_view2.gz')

    train_set_x1, train_set_y1 = data1[0]
    valid_set_x1, valid_set_y1 = data1[1]
    test_set_x1, test_set_y1 = data1[2]

    train_set_x2, train_set_y2 = data2[0]
    valid_set_x2, valid_set_y2 = data2[1]
    test_set_x2, test_set_y2 = data2[2]

    input1 = Input(shape=(input_shape1, ), name='input1')
    input2 = Input(shape=(input_shape1, ), name='input2')

    expert_index = 0
    activation_model = 'sigmoid'
    dense1_1 = Dense(1024, activation=activation_model, name='view_1_1')(input1)
    dense1_2 = Dense(1024, activation=activation_model, name='view_1_2')(dense1_1)
    dense1_3 = Dense(1024, activation=activation_model,  name='view_1_3')(dense1_2)
    output1 = Dense(10, activation='linear', name='view_1_4')(dense1_3)

    dense2_1 = Dense(1024, activation=activation_model,  name='view_2_1')(input2)
    dense2_2 = Dense(1024, activation=activation_model,  name='view_2_2')(dense2_1)
    dense2_3 = Dense(1024, activation=activation_model, name='view_2_3')(dense2_2)
    output2 = Dense(10, activation='linear', name='view_2_4')(dense2_3)

    shared_layer = concatenate([output1, output2], name='shared_layer')

    cca_layer = CCA(1, name='cca_layer')(shared_layer)

    model = Model(inputs=[input1, input2], outputs=cca_layer)
    model.compile(optimizer='rmsprop', loss=constant_loss, metrics=[mean_pred])
    model.fit([train_set_x1, train_set_x2], np.zeros(len(train_set_x1)),
              batch_size=batch_size, epochs=epoch_num, shuffle=True, verbose=1,
              validation_data=([valid_set_x1, valid_set_x2], np.zeros(len(valid_set_x1))))


    # evaluation for view_1
    new_data    = []
    expert_data = []
    for i in range(1):
        expert_result = []
        current_dcca = Model(input=model.input, output=model.get_layer(name='shared_layer').output)

        current_expert_data = []
        for k in range(3):
            a = data1[k][0]
            b = data2[k][0]
            pred_out = current_dcca.predict([a, b])
            r = int(pred_out.shape[1] / 2)
            current_expert_data.append([pred_out[:, :r], pred_out[:, r:]])

        if i == 0:
            expert_data = current_expert_data
        else:
            expert_data = np.add(expert_data, current_expert_data)

    new_data.append([expert_data[0][0]/50000, expert_data[0][1]/50000, data1[0][1]])
    new_data.append([expert_data[1][0]/10000, expert_data[1][1]/10000, data1[1][1]])
    new_data.append([expert_data[2][0]/10000, expert_data[2][1]/10000, data1[2][1]])


    [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
    print("current accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    print("current accuracy on view 1 (test data) is:", test_acc * 100.0)

    print("training ended!")

