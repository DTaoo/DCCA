from keras import backend as T
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np


def my_eigen(x):
    return np.linalg.eigh(x)

def my_svd(x):
    return np.linalg.svd(x, compute_uv=False)

class CCA(Layer):
    '''CCA layer is used compute the CCA objective

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        output_dim: output dimension, default 1, i.e., correlation coefficient
        use_all_singular_value: if use the top-k singular values
        cca_space_dim: the number of singular values, i.e., k

    '''

    def __init__(self, output_dim=1, use_all_singular_values=True, cca_space_dim=10, **kwargs):
        self.output_dim = output_dim
        self.cca_space_dim = cca_space_dim
        self.use_all_singular_values = use_all_singular_values
        super(CCA, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(CCA, self).build(input_shape)

    def call(self, x):
        r1 = tf.constant([1e-4])
        r2 = tf.constant([1e-4])
        eps = tf.constant([1e-12])
        o1 = o2 = tf.shape(x)[1] // 2

        H1 = T.transpose(x[:, 0:o1])
        H2 = T.transpose(x[:, o1:o1 + o2])

        one = tf.constant([1.0])
        m = tf.shape(H1)[1]
        m_float = tf.cast(m, 'float')

        # minus the mean value
        partition = tf.divide(one, m_float)
        H1bar = H1 - partition * tf.matmul(H1, tf.ones([m, m]))
        H2bar = H2 - partition * tf.matmul(H2, tf.ones([m, m]))

        # calculate the auto-covariance and cross-covariance
        partition2 = tf.divide(one, (m_float - 1))
        SigmaHat12 = partition2 * tf.matmul(H1bar, tf.transpose(H2bar))
        SigmaHat11 = partition2 * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(o1)
        SigmaHat22 = partition2 * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(o2)

        # calculate the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = tf.py_func(my_eigen, [SigmaHat11], [tf.float32, tf.float32])
        [D2, V2] = tf.py_func(my_eigen, [SigmaHat22], [tf.float32, tf.float32])

        # for stability
        D1_indices = tf.where(D1 > eps)
        D1_indices = tf.squeeze(D1_indices)
        V1 = tf.gather(V1, D1_indices)
        D1 = tf.gather(D1, D1_indices)

        D2_indices = tf.where(D2 > eps)
        D2_indices = tf.squeeze(D2_indices)
        V2 = tf.gather(V2, D2_indices)
        D2 = tf.gather(D2, D2_indices)

        pow_value = tf.constant([-0.5])
        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(tf.pow(D1, pow_value))), tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(tf.pow(D2, pow_value))), tf.transpose(V2))

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = tf.trace(T.sqrt(tf.matmul(tf.transpose(Tval), Tval)))
        else:
            # just the top outdim_size singular values are used
            TT = tf.matmul(tf.transpose(Tval), Tval)
            U, V = tf.self_adjoint_eig(TT)
            U_sort, _ = tf.nn.top_k(U, self.cca_space_dim)
            corr = T.sum(T.sqrt(U_sort))

        return -corr

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'cca_dim': self.cca_dim,
            'use_all_singular_values': self.use_all_singular_values,
        }
        base_config = super(CCA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))