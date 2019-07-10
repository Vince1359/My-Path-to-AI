# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: 1_classify_images.py
# 创建时间: 2019/7/8 下午7:59

import tensorflow as tf
from tensorflow import keras


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def prepare_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    train_images, train_labels, test_images, test_labels = prepare_data()
    model = build_model()
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}, Test loss: {}'.format(test_acc, test_loss))


if __name__ == '__main__':
    main()
