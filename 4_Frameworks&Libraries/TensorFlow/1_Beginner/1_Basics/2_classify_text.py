# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: 2_classify_text.py
# 创建时间: 2019/7/8 下午7:59

import tensorflow as tf
from tensorflow import keras

import tensorflow_hub as hub
import tensorflow_datasets as tfds


def prepare_data():
    train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

    (train_data, validation_data), test_data = tfds.load(
        name="imdb_reviews",
        split=(train_validation_split, tfds.Split.TEST),
        as_supervised=True)

    return train_data, validation_data, test_data


def build_model():
    embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)

    model = keras.Sequential()
    model.add(hub_layer)
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    train_data, validation_data, test_data = prepare_data()
    model = build_model()

    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=20,
                        validation_data=validation_data.batch(512),
                        verbose=1)

    results = model.evaluate(test_data.batch(512), verbose=0)
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))
        print('{}: {:3f}'.format(name, value))


if __name__ == '__main__':
    main()
