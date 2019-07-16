# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: 5_overfitting_underfitting.py
# 创建时间: 2019/7/8 下午8:00

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


def prepare_data():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

    return train_data, train_labels, test_data, test_labels


def build_baseline_model():
    baseline_model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    baseline_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'binary_crossentropy'])

    baseline_model.summary()
    return baseline_model


def build_smaller_model():
    smaller_model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    smaller_model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy', 'binary_crossentropy'])
    smaller_model.summary()
    return smaller_model


def build_bigger_model():
    bigger_model = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    bigger_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy', 'binary_crossentropy'])
    bigger_model.summary()
    return bigger_model


def build_l2_model():
    l2_model = keras.models.Sequential([
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    l2_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])
    l2_model.summary()
    return l2_model


def build_dropout_model():
    dropout_model = keras.models.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    dropout_model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy', 'binary_crossentropy'])
    dropout_model.summary()
    return dropout_model


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = prepare_data()
    baseline_model = build_baseline_model()
    smaller_model = build_smaller_model()
    bigger_model = build_bigger_model()
    l2_model = build_l2_model()
    dropout_model = build_dropout_model()

    baseline_history = baseline_model.fit(train_data,
                                          train_labels,
                                          epochs=20,
                                          batch_size=512,
                                          validation_data=(test_data, test_labels),
                                          verbose=2)
    smaller_history = smaller_model.fit(train_data,
                                        train_labels,
                                        epochs=20,
                                        batch_size=512,
                                        validation_data=(test_data, test_labels),
                                        verbose=2)
    bigger_history = bigger_model.fit(train_data, train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
    l2_history = l2_model.fit(train_data, train_labels,
                              epochs=20,
                              batch_size=512,
                              validation_data=(test_data, test_labels),
                              verbose=2)
    dropout_history = dropout_model.fit(train_data, train_labels,
                                        epochs=20,
                                        batch_size=512,
                                        validation_data=(test_data, test_labels),
                                        verbose=2)

    plot_history([('baseline', baseline_history),
                  ('smaller', smaller_history),
                  ('bigger', bigger_history),
                  ('l2', l2_history),
                  ('dropout', dropout_history)])


if __name__ == '__main__':
    main()
