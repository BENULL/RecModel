#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/6 下午5:04
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.feature_column.dense_features_v2 import DenseFeatures
from tensorflow.python.keras.layers import concatenate

from WDL.modules import Linear, DNN


class WideDeep(Model):

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                 l2_linear_reg=1e-6, dnn_activation='relu', dnn_dropout=0., ):
        """
        Wide&Deep
        :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param dnn_hidden_units: A list. Neural network hidden units.
        :param dnn_activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param l2_linear_reg: A scalar. The regularizer of Linear.
        """
        super(WideDeep, self).__init__()
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.wide_dense_feature = DenseFeatures(self.linear_feature_columns)
        self.deep_dense_feature = DenseFeatures(self.dnn_feature_columns)
        self.linear = Linear(use_bias=True, l2_linear_reg=l2_linear_reg)
        self.dnn_network = DNN(dnn_hidden_units, dnn_activation, dnn_dropout)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # Wide
        wide_out = self.wide_dense_feature(inputs)
        # wide_out = self.linear(wide_inputs)
        # Deep
        deep_inputs = self.deep_dense_feature(inputs)
        deep_out = self.dnn_network(deep_inputs)
        both = concatenate([deep_out, wide_out])
        # Out
        outputs = self.output_layer(both)
        return outputs
