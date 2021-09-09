#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/6 下午5:09
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.regularizers import l2


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
        self.w = self.add_weight(name="w",
                                 shape=(int(input_shape[-1]), 1),
                                 regularizer=l2(self.l2_reg_linear),
                                 trainable=True)
        super(Linear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        linear_logit = tf.tensordot(inputs, self.w, axes=(-1, 0))
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
