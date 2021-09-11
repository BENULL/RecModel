#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/9/7 上午12:09
"""

import tensorflow as tf
from tensorflow.keras.utils import get_file


# load sample as tf dataset
def get_dataset(file_path, batchsize):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batchsize,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


def load_ml_dataset(batchsize):
    # Training samples path
    training_samples_file_path = get_file("trainingSamples.csv", "file://../dataset/movielens/trainingSamples.csv")
    # Test samples path
    test_samples_file_path = get_file("testSamples.csv", "file://../dataset/movielens/testSamples.csv")

    # split as test dataset and training dataset
    train_dataset = get_dataset(training_samples_file_path, batchsize)
    test_dataset = get_dataset(test_samples_file_path, batchsize)
    return train_dataset, test_dataset


def create_ml_dataset(batchsize=12):
    train_dataset, test_dataset = load_ml_dataset(batchsize)

    # movie id embedding feature
    movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
    movie_ind_col = tf.feature_column.indicator_column(movie_col)  # movie id indicator columns

    # user id embedding feature
    user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)
    user_ind_col = tf.feature_column.indicator_column(user_col)  # user id indicator columns

    # genre features vocabulary
    genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
                   'Sci-Fi', 'Drama', 'Thriller',
                   'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

    # user genre embedding feature
    user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                               vocabulary_list=genre_vocab)
    user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, 10)
    user_genre_ind_col = tf.feature_column.indicator_column(user_genre_col)  # user genre indicator columns

    # item genre embedding feature
    item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                               vocabulary_list=genre_vocab)
    item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, 10)
    item_genre_ind_col = tf.feature_column.indicator_column(item_genre_col)  # item genre indicator columns

    # deep features
    dense_feature_columns = [tf.feature_column.numeric_column('releaseYear'),
                             tf.feature_column.numeric_column('movieRatingCount'),
                             tf.feature_column.numeric_column('movieAvgRating'),
                             tf.feature_column.numeric_column('movieRatingStddev'),
                             tf.feature_column.numeric_column('userRatingCount'),
                             tf.feature_column.numeric_column('userAvgRating'),
                             tf.feature_column.numeric_column('userRatingStddev')]

    sparse_feature_columns = [
        movie_emb_col, user_emb_col, user_genre_emb_col, item_genre_emb_col
    ]

    # fm first-order term columns: without embedding and concatenate to the output layer directly
    # fm_first_order_columns = [movie_ind_col, user_ind_col, user_genre_ind_col, item_genre_ind_col]

    inputs = {
        'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
        'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
        'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
        'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
        'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
        'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
        'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

        'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
        'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
        'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

        'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
        'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
        'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
        'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
        'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
        'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
        'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
        'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
    }

    feature_columns = dict(sparse_feature_columns=sparse_feature_columns,
                           dense_feature_columns=dense_feature_columns, )
    return feature_columns, inputs, train_dataset, test_dataset
