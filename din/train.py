#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/6 下午11:41
"""
import tensorflow as tf
from din.model import DIN
from din.dataprocess_ml import create_ml_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # ========================= Hyper Parameters =======================
    dnn_dropout = 0.5
    dnn_hidden_units = [128, 64]
    attn_hidden_units = [32]
    batch_size = 12
    epochs = 1

    # ========================== Create dataset =======================
    feature_columns, inputs, train_dataset, test_dataset = create_ml_dataset(batch_size)
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        model = DIN(feature_columns, attn_hidden_units, dnn_hidden_units)

        # ============================Compile============================

        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

        # ============================model checkpoint======================
        # check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
        #                                                 verbose=1, period=5)

    # ==============================Fit==============================
    model.fit(
        train_dataset,
        epochs=epochs,
    )

    model.summary()
    # ===========================Test==============================
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
    print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                       test_roc_auc, test_pr_auc))

    # print some predict results
    predictions = model.predict(test_dataset)
    for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ",
              ("Good Rating" if bool(goodRating) else "Bad Rating"))
