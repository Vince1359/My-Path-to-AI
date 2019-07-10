# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: 4_regression.py
# 创建时间: 2019/7/8 下午8:00

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def norm(x, train_dataset):
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    return (x - train_stats['mean']) / train_stats['std']


def prepare_data():
    dataset_path = keras.utils.get_file(
        "auto-mpg.data",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')

    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    normed_train_data = norm(train_dataset, train_dataset)
    normed_test_data = norm(test_dataset, train_dataset)

    return normed_train_data, train_labels, normed_test_data, test_labels


def build_model(train_dataset):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


def main():
    normed_train_data, train_labels, normed_test_data, test_labels = prepare_data()
    model = build_model(normed_train_data)

    epochs = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=epochs, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
    print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))
    plot_history(history)

    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])


if __name__ == '__main__':
    main()
