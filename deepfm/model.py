#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/11 下午4:36
"""
import tensorflow as tf
from tensorflow.python.keras.feature_column.dense_features_v2 import DenseFeatures
from tensorflow.python.keras.layers import Dense, concatenate, add, Reshape
from tensorflow.python.keras.models import Model

from deepfm.modules import FM, Linear, DNN


class DeepFM(Model):

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128, 64),
                 l2_linear_reg=1e-6, dnn_activation='relu', dnn_dropout=0., ):
        super(DeepFM, self).__init__()
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.embedding = DenseFeatures(self.linear_feature_columns)
        self.dense_feature = DenseFeatures(self.dnn_feature_columns)
        self.linear = Linear(use_bias=True, l2_linear_reg=l2_linear_reg)
        self.reshape = Reshape((len(self.linear_feature_columns), -1))
        self.fm = FM()
        self.dnn_network = DNN(dnn_hidden_units, dnn_activation, dnn_dropout)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # first order term
        embeddings = self.embedding(inputs)
        first_order = self.linear(embeddings)

        # second order term
        embed_inputs = self.reshape(embeddings)
        second_order = self.fm(embed_inputs)

        # dnn term
        dnn_inputs = self.dense_feature(inputs)
        dnn_inputs = concatenate([embeddings, dnn_inputs])
        dnn_out = self.dnn_network(dnn_inputs)

        # out
        both = concatenate([first_order, second_order, dnn_out])
        outputs = self.output_layer(both)
        return outputs

