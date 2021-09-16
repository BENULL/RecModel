#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/15 下午1:53
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense, Dropout, PReLU, BatchNormalization


class AttentionLayer(Layer):
    def __init__(self, keys_dim, att_hidden_units, activation='prelu'):
        super(AttentionLayer, self).__init__()
        self.keys_dim = keys_dim
        self.att_dense = [Dense(units=unit, activation=PReLU() if activation == 'prelu' else Dice())
                          for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs, **kwargs):
        # query: candidate item  (None, d), d is the dimension of embedding
        # key: hist items  (None, lens_k, d)
        q, k = inputs
        #           (None, d) => (None, 1, d)
        q = tf.tile(tf.expand_dims(q, 1), [1, tf.shape(k)[1], 1])  # (None, lens_k, d)
        din_all = tf.concat([q, k, q-k, q*k], axis=-1)

        # dense
        outputs = None
        for dense in self.att_dense:
            outputs = dense(din_all)
        outputs = tf.transpose(self.att_final_dense(outputs), [0, 2, 1])  # (None, 1, lens_k)

        # key_masks = tf.sequence_mask(keys_length, max(keys_length), dtype=tf.bool)  # (None, lens_k)
        # key_masks = tf.expand_dims(key_masks, 1)  # (None, 1, lens_k)
        # paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, 1, lens_k)
        # outputs = tf.where(key_masks, outputs, paddings)  # (None, lens_k, 1, max(lens_k))
        # outputs = outputs / (self.keys_dim ** 0.5)  # (None, lens_k, 1, max(lens_k))

        outputs = tf.keras.activations.sigmoid(outputs)  # (None, 1, lens_k)

        outputs = tf.matmul(outputs, k)  # (None, 1, lens_k) matmul (None, lens_k, d)   = (None, 1, d)
        outputs = tf.squeeze(outputs, axis=1)  # (None,  d)
        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, inputs, **kwargs):
        x = inputs
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x


class DNN(Layer):

    def __init__(self, dnn_hidden_units, dnn_activation='prelu', dnn_dropout=0.):
        """
        Deep Neural Network
        :param dnn_hidden_units: A list. Neural network hidden units.
        :param dnn_activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=PReLU() if dnn_activation == 'prelu' else Dice()) 
                            for unit in dnn_hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


