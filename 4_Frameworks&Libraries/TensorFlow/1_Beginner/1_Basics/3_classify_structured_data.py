# -*- coding: utf-8 -*-

# 作者: Vince1359
# 文件名: 3_classify_structured_data.py
# 创建时间: 2019/7/8 下午8:00

import pandas as pd

import tensorflow as tf
from tensorflow import keras

from tensorflow import feature_column
from sklearn.model_selection import train_test_split


def df_to_dataset(data_frame, shuffle=True, batch_size=32):
    data_frame = data_frame.copy()
    labels = data_frame.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(data_frame), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_frame))
    ds = ds.batch(batch_size)
    return ds


def prepare_data():
    url = 'https://storage.googleapis.com/applied-dl/heart.csv'
    data_frame = pd.read_csv(url)
    data_frame.head()

    train, test = train_test_split(data_frame, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds


def build_model():
    feature_columns = []

    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(feature_column.numeric_column(header))

    age = feature_column.numeric_column("age")
    age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    thal = feature_column.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    thal_embedding = feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    crossed_feature = feature_column.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

    feature_layer = keras.layers.DenseFeatures(feature_columns)

    model = tf.keras.Sequential([
        feature_layer,
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)

    return model


def main():
    train_ds, val_ds, test_ds = prepare_data()
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=5)
    loss, accuracy = model.evaluate(test_ds)
    print('Accuracy: {}, Loss: {}'.format(accuracy, loss))


if __name__ == '__main__':
    main()
