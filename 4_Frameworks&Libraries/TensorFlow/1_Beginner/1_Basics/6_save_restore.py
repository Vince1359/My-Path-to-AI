# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: 6_save_restore.py
# 创建时间: 2019/7/8 下午8:00

import tensorflow as tf
from tensorflow import keras


def prepare_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    return train_images, train_labels, test_images, test_labels


def build_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def create_checkpoint_callback(checkpoint_path, period=1):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=period)
    return cp_callback


def main():
    train_images, train_labels, test_images, test_labels = prepare_data()
    model = build_model()

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[create_checkpoint_callback('training_1/cp.ckpt', period=5)])


if __name__ == '__main__':
    main()
