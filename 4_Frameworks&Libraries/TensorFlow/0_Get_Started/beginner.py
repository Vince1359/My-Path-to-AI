# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: beginner.py
# 创建时间: 2019/7/8 下午5:19

import tensorflow as tf


def prepare_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    x_train, y_train, x_test, y_test = prepare_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
