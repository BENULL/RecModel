#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/15 下午1:50
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.core import Dense
from tensorflow.python.layers.normalization import BatchNormalization

from din.modules import AttentionLayer, DNN


class DIN(Model):

    def __init__(self, feature_columns, att_hidden_units=(80, 40),
                 dnn_hidden_units=(80, 40), att_activation='prelu', dnn_activation='prelu', dnn_dropout=0.,):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param dnn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param dnn_activation: A String. Prelu or Dice.
        :param dnn_dropout: A scalar. The number of Dropout.
        """
        super(DIN, self).__init__()
        self.candidate_layer = tf.keras.layers.DenseFeatures(feature_columns['candidate_col'])
        self.user_behaviors_layer = tf.keras.layers.DenseFeatures(feature_columns['behavior_col'])
        self.behaviors_nums = len(feature_columns['behavior_col'])
        self.user_profile_layer = tf.keras.layers.DenseFeatures(feature_columns['user_profile'])
        self.context_features_layer = tf.keras.layers.DenseFeatures(feature_columns['context_features'])

        self.attention_layer = AttentionLayer(len(feature_columns['behavior_col']),
                                              att_hidden_units, activation=att_activation)

        self.bn = BatchNormalization(trainable=True)
        self.dnn_network = DNN(dnn_hidden_units, dnn_activation, dnn_dropout)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        candidate_emb = self.candidate_layer(inputs)  # (None, d)
        user_behaviors_emb = self.user_behaviors_layer(inputs)  # (None, lens_k*d)
        user_behaviors_emb = Reshape((self.behaviors_nums, -1))(user_behaviors_emb)
        user_profile_emb = self.user_profile_layer(inputs)
        context_features_emb = self.context_features_layer(inputs)

        # attention   (q, k, keys_length)
        activation_unit = self.attention_layer((candidate_emb, user_behaviors_emb,))
        all_input = tf.concat([activation_unit, user_profile_emb, context_features_emb], axis=-1)

        all_out = self.bn(all_input)
        dnn_out = self.dnn_network(all_out)
        outputs = self.output_layer(dnn_out)
        return outputs






