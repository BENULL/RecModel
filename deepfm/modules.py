#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/11 下午4:34
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.python.keras import backend as K


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        return cross_term

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return None, 1


class Linear(Layer):
    def __init__(self, use_bias=False, l2_linear_reg=1e-6):
        super(Linear, self).__init__()
        self.use_bias = use_bias
        self.l2_reg_linear = l2_linear_reg

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        linear_logit = tf.reduce_sum(inputs, axis=-1, keepdims=True)
        if self.use_bias:
            linear_logit += self.bias
        return linear_logit


class DNN(Layer):

    def __init__(self, dnn_hidden_units, dnn_activation='relu', dnn_dropout=0.):
        """
        Deep Neural Network
        :param dnn_hidden_units: A list. Neural network hidden units.
        :param dnn_activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=dnn_activation) for unit in dnn_hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
