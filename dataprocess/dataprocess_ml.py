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

def create_ml_dataset(batchsize = 12):
    train_dataset, test_dataset = load_ml_dataset(batchsize)

    # genre features vocabulary
    genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
                   'Sci-Fi', 'Drama', 'Thriller',
                   'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

    GENRE_FEATURES = {
        'userGenre1': genre_vocab,
        'userGenre2': genre_vocab,
        'userGenre3': genre_vocab,
        'userGenre4': genre_vocab,
        'userGenre5': genre_vocab,
        'movieGenre1': genre_vocab,
        'movieGenre2': genre_vocab,
        'movieGenre3': genre_vocab
    }

    # all categorical features
    categorical_columns = []
    for feature, vocab in GENRE_FEATURES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        emb_col = tf.feature_column.embedding_column(cat_col, 10)
        categorical_columns.append(emb_col)

    # movie id embedding feature
    movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
    categorical_columns.append(movie_emb_col)

    # user id embedding feature
    user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)
    categorical_columns.append(user_emb_col)

    # all numerical features
    numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                         tf.feature_column.numeric_column('movieRatingCount'),
                         tf.feature_column.numeric_column('movieAvgRating'),
                         tf.feature_column.numeric_column('movieRatingStddev'),
                         tf.feature_column.numeric_column('userRatingCount'),
                         tf.feature_column.numeric_column('userAvgRating'),
                         tf.feature_column.numeric_column('userRatingStddev')]

    # cross feature between current movie and user historical movie
    rated_movie = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
    crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([movie_col, rated_movie], 10000))

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

    feature_columns = dict(numerical_columns=numerical_columns,
                           categorical_columns=categorical_columns,
                           crossed_columns=crossed_feature)
    return feature_columns, inputs, train_dataset, test_dataset

if __name__ == '__main__':
    feature_columns, inputs, train_dataset, test_dataset = create_ml_dataset()
    print(feature_columns)
